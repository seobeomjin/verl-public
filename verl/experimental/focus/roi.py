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
"""Region-of-interest proposal and selection on an object relevance map (FOCUS, Sec. 3.3 / App. A.5).

Boxes are ``(x0, y0, x1, y1)`` with exclusive ends, in relevance-map grid cells unless noted otherwise.
"""

import re
from dataclasses import dataclass

import torch

Box = tuple[int, int, int, int]


@dataclass
class ROI:
    box: Box
    relevance: float
    confidence: float | None = None


def extract_anchors(rel_map: torch.Tensor, k: int, min_dist: float) -> list[tuple[int, int]]:
    """Top-k ``(y, x)`` peaks of ``rel_map``, greedily skipping peaks closer than ``min_dist`` to a kept one."""
    w = rel_map.shape[1]
    anchors: list[tuple[int, int]] = []
    for flat in torch.argsort(rel_map.flatten(), descending=True).tolist():
        y, x = divmod(flat, w)
        if all((y - ay) ** 2 + (x - ax) ** 2 >= min_dist**2 for ay, ax in anchors):
            anchors.append((y, x))
            if len(anchors) == k:
                break
    return anchors


def _square_box(cy: int, cx: int, side: int, grid_hw: tuple[int, int]) -> Box:
    """Square box of ``side`` centered on ``(cy, cx)``, shifted (not clipped) to stay inside the grid."""
    h, w = grid_hw
    side_h, side_w = min(side, h), min(side, w)
    y0 = min(max(cy - side_h // 2, 0), h - side_h)
    x0 = min(max(cx - side_w // 2, 0), w - side_w)
    return x0, y0, x0 + side_w, y0 + side_h


def _box_mean(rel_map: torch.Tensor, box: Box) -> float:
    x0, y0, x1, y1 = box
    return rel_map[y0:y1, x0:x1].mean().item()


def expand_roi(rel_map: torch.Tensor, anchor: tuple[int, int], s_min: int, s_max: int, expand_thresh: float) -> Box:
    """Grow a square ROI around ``anchor`` from ``s_min`` up to ``s_max`` cells.

    Growth stops once the mean relevance inside the grown box would fall below ``expand_thresh`` times
    the uniform level ``1 / (h * w)``, so the threshold does not depend on the map's resolution.
    """
    grid_hw = tuple(rel_map.shape)
    floor = expand_thresh * rel_map.sum().item() / rel_map.numel()
    box = _square_box(*anchor, s_min, grid_hw)
    side = s_min
    while side + 2 <= s_max:
        grown = _square_box(*anchor, side + 2, grid_hw)
        if grown == box or _box_mean(rel_map, grown) < floor:
            break
        box, side = grown, side + 2
    return box


def box_iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(rois: list[ROI], iou_thresh: float) -> list[ROI]:
    """Non-maximum suppression by relevance; a low threshold promotes spatial diversity."""
    kept: list[ROI] = []
    for roi in sorted(rois, key=lambda r: r.relevance, reverse=True):
        if all(box_iou(roi.box, k.box) <= iou_thresh for k in kept):
            kept.append(roi)
    return kept


def propose_rois(
    rel_map: torch.Tensor,
    k: int = 30,
    s_min: int = 3,
    s_max: int = 5,
    s_dist: float = 2,
    expand_thresh: float = 1.0,
    nms_iou: float = 0.3,
) -> list[ROI]:
    """Anchors -> expanded ROIs -> NMS, ranked by the relevance at each anchor."""
    rois = [
        ROI(box=expand_roi(rel_map, a, s_min, s_max, expand_thresh), relevance=rel_map[a].item())
        for a in extract_anchors(rel_map, k, s_dist)
    ]
    return nms(rois, nms_iou)


def grid_box_to_pixels(box: Box, grid_hw: tuple[int, int], image_size: tuple[int, int]) -> Box:
    """Map a grid box to pixel coordinates of an image of ``(width, height)``."""
    h, w = grid_hw
    width, height = image_size
    sx, sy = width / w, height / h
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        min(width, int(round(x1 * sx))),
        min(height, int(round(y1 * sy))),
    )


def scale_box(box: Box, from_size: tuple[int, int], to_size: tuple[int, int]) -> Box:
    """Rescale a pixel box between two resolutions of the same image (sizes are ``(width, height)``)."""
    sx, sy = to_size[0] / from_size[0], to_size[1] / from_size[1]
    x0, y0, x1, y1 = box
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        min(to_size[0], int(round(x1 * sx))),
        min(to_size[1], int(round(y1 * sy))),
    )


def existence_confidence(logit_yes: float, logit_no: float) -> float:
    """``c_yes = 2 * (softmax([l_yes, l_no])_yes - 0.5)`` in ``[-1, 1]`` (Eq. 5)."""
    p_yes = torch.softmax(torch.tensor([logit_yes, logit_no], dtype=torch.float32), dim=0)[0].item()
    return 2 * (p_yes - 0.5)


_TYPE2_PATTERN = re.compile(r"\b(how many|number of|count)\b", re.IGNORECASE)


def is_type2_question(question: str) -> bool:
    """Keyword heuristic for questions about multiple instances of an object type (e.g. counting)."""
    return bool(_TYPE2_PATTERN.search(question))


def select_rois(ranked: list[ROI], type2: bool, t_type2: float) -> list[ROI]:
    """Pick the final ROIs of one target from confidence-ranked candidates.

    Type-1: the single most confident ROI. Type-2: every ROI above ``t_type2`` (falling back to the
    top-1), with overlapping ones merged into their union.
    """
    if not ranked:
        return []
    if not type2:
        return ranked[:1]
    chosen = [r for r in ranked if r.confidence is not None and r.confidence > t_type2] or ranked[:1]
    merged: list[ROI] = []
    for roi in chosen:
        for i, m in enumerate(merged):
            if box_iou(roi.box, m.box) > 0:
                merged[i] = ROI(
                    box=(
                        min(m.box[0], roi.box[0]),
                        min(m.box[1], roi.box[1]),
                        max(m.box[2], roi.box[2]),
                        max(m.box[3], roi.box[3]),
                    ),
                    relevance=max(m.relevance, roi.relevance),
                    confidence=max(m.confidence or -1.0, roi.confidence or -1.0),
                )
                break
        else:
            merged.append(roi)
    return merged
