# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model-agnostic FOCUS pipeline: target extraction -> relevance map -> ROI proposal -> existence ranking."""

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import torch
from PIL import Image

from verl.experimental.focus.prompts import parse_targets
from verl.experimental.focus.relevance import compute_relevance_map
from verl.experimental.focus.roi import (
    ROI,
    existence_confidence,
    grid_box_to_pixels,
    is_type2_question,
    propose_rois,
    select_rois,
)


@dataclass
class FocusConfig:
    """FOCUS hyperparameters (defaults follow Tab. 7 of the paper, LLaVA-OneVision column).

    ``s_min`` / ``s_max`` / ``s_dist`` are in relevance-map cells. Pick ``s_max`` so that a crop spans
    roughly 1-2x the vision encoder's base resolution in the original image.
    """

    k: int = 30
    s_min: int = 3
    s_max: int = 5
    s_dist: float = 2
    expand_thresh: float = 1.0
    nms_iou: float = 0.3
    n_steps: int = 4
    t_type2: float = 0.5
    smooth_sigma: float = 1.0
    layer_frac: tuple[float, float] = (0.5, 1.0)
    max_targets: int = 3
    max_crops: int = 4

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FocusConfig":
        d = dict(d or {})
        if "layer_frac" in d:
            d["layer_frac"] = tuple(d["layer_frac"])
        return cls(**d)


@dataclass
class FocusCrop:
    target: str
    box: tuple[int, int, int, int]  # pixels in the original image, (x0, y0, x1, y1)
    relevance: float
    confidence: float | None


@dataclass
class FocusResult:
    targets: list[str] = field(default_factory=list)
    type2: bool = False
    crops: list[FocusCrop] = field(default_factory=list)
    num_forward_passes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValueFeatures:
    values: torch.Tensor | list[torch.Tensor]  # [L, N, D] or per-layer [N, D_l] value features
    target_idx: list[int]
    visual_idx: list[int]
    grid_hw: tuple[int, int]


class FocusBackend(Protocol):
    """The three model calls FOCUS needs. Implemented by ``FocusWorker`` with a frozen HF model."""

    def extract_targets(self, question: str) -> str: ...

    def value_features(self, image: Image.Image, target: str) -> ValueFeatures | None: ...

    def existence_logits(self, images: list[Image.Image], target: str) -> list[tuple[float, float]]: ...


def run_focus(
    backend: FocusBackend,
    image: Image.Image,
    question: str,
    config: FocusConfig,
    targets: list[str] | None = None,
) -> FocusResult:
    """Run FOCUS steps I-V on one image/question and return the selected crops (pixel boxes)."""
    result = FocusResult(type2=is_type2_question(question))
    if not targets:
        targets = parse_targets(backend.extract_targets(question), config.max_targets)
        result.num_forward_passes += 1
    result.targets = list(targets)[: config.max_targets]

    for target in result.targets:
        feats = backend.value_features(image, target)
        result.num_forward_passes += 1
        if feats is None or not feats.target_idx:
            continue
        rel_map = compute_relevance_map(
            feats.values, feats.target_idx, feats.visual_idx, feats.grid_hw, config.smooth_sigma
        )
        candidates = propose_rois(
            rel_map, config.k, config.s_min, config.s_max, config.s_dist, config.expand_thresh, config.nms_iou
        )
        # Type-2 questions look at every candidate (not bounded by n_steps), as in the paper.
        candidates = candidates if result.type2 else candidates[: config.n_steps]
        pixel_boxes = [grid_box_to_pixels(r.box, feats.grid_hw, image.size) for r in candidates]
        logits = backend.existence_logits([image.crop(b) for b in pixel_boxes], target)
        result.num_forward_passes += len(pixel_boxes)
        ranked = [
            ROI(box=b, relevance=r.relevance, confidence=existence_confidence(*lg))
            for r, b, lg in zip(candidates, pixel_boxes, logits, strict=True)
        ]
        ranked.sort(key=lambda r: r.confidence, reverse=True)
        for roi in select_rois(ranked, result.type2, config.t_type2):
            result.crops.append(FocusCrop(target, roi.box, roi.relevance, roi.confidence))

    result.crops = result.crops[: config.max_crops]
    return result
