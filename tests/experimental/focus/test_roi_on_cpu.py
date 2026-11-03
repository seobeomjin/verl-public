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
"""CPU tests for FOCUS ROI proposal / selection, prompts, and the end-to-end pipeline with a fake model."""

import pytest
import torch
from PIL import Image

from verl.experimental.focus.pipeline import FocusConfig, ValueFeatures, run_focus
from verl.experimental.focus.prompts import build_target_extraction_prompt, parse_targets
from verl.experimental.focus.roi import (
    ROI,
    box_iou,
    existence_confidence,
    expand_roi,
    extract_anchors,
    grid_box_to_pixels,
    is_type2_question,
    nms,
    propose_rois,
    scale_box,
    select_rois,
)


def _blob_map(h, w, centers, radius=1):
    m = torch.full((h, w), 0.01)
    for cy, cx in centers:
        m[max(0, cy - radius) : cy + radius + 1, max(0, cx - radius) : cx + radius + 1] += 1.0
        m[cy, cx] += 1.0
    return m / m.sum()


def test_extract_anchors_respects_min_distance():
    m = _blob_map(10, 10, [(2, 2), (7, 7)])
    anchors = extract_anchors(m, k=5, min_dist=3)
    assert anchors[:2] in ([(2, 2), (7, 7)], [(7, 7), (2, 2)])
    for i, a in enumerate(anchors):
        for b in anchors[i + 1 :]:
            assert (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 >= 9


def test_expand_roi_grows_over_relevant_region_and_stops():
    m = _blob_map(12, 12, [(6, 6)], radius=2)
    # Mean relevance relative to uniform: 5x5 -> 5.4x, 7x7 -> 2.9x, 9x9 -> 1.8x.
    assert expand_roi(m, (6, 6), s_min=3, s_max=9, expand_thresh=3.0) == (4, 4, 9, 9)
    assert expand_roi(m, (6, 6), s_min=3, s_max=9, expand_thresh=2.0) == (3, 3, 10, 10)
    # s_max caps the growth.
    assert expand_roi(m, (6, 6), s_min=3, s_max=3, expand_thresh=0.0) == (5, 5, 8, 8)


def test_expand_roi_shifts_box_inside_border():
    m = _blob_map(6, 8, [(0, 0)])
    x0, y0, x1, y1 = expand_roi(m, (0, 0), s_min=3, s_max=3, expand_thresh=0.0)
    assert (x0, y0, x1, y1) == (0, 0, 3, 3)
    # Boxes never exceed a grid smaller than s_max.
    assert expand_roi(torch.ones(2, 2), (1, 1), 3, 9, 0.0) == (0, 0, 2, 2)


def test_nms_and_iou():
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert box_iou((0, 0, 2, 2), (2, 2, 4, 4)) == 0.0
    rois = [ROI((0, 0, 3, 3), 0.9), ROI((1, 0, 4, 3), 0.8), ROI((6, 6, 9, 9), 0.5)]
    assert [r.relevance for r in nms(rois, 0.3)] == [0.9, 0.5]


def test_propose_rois_ranks_by_relevance():
    m = _blob_map(12, 12, [(2, 9)])
    m[8, 3] += 0.5 * m.max()
    rois = propose_rois(m, k=10, s_min=3, s_max=5, s_dist=2)
    assert rois[0].box[0] <= 9 < rois[0].box[2] and rois[0].box[1] <= 2 < rois[0].box[3]
    assert [r.relevance for r in rois] == sorted((r.relevance for r in rois), reverse=True)


def test_box_coordinate_mappings():
    # 19x28 grid of a 2246x1582 image (Gemma 4, 560 soft tokens).
    assert grid_box_to_pixels((0, 0, 28, 19), (19, 28), (2246, 1582)) == (0, 0, 2246, 1582)
    assert grid_box_to_pixels((14, 5, 19, 10), (19, 28), (2240, 1520)) == (1120, 400, 1520, 800)
    assert scale_box((100, 50, 200, 150), (1000, 500), (4000, 2000)) == (400, 200, 800, 600)
    assert scale_box((0, 0, 1000, 500), (1000, 500), (999, 499)) == (0, 0, 999, 499)


def test_existence_confidence():
    assert existence_confidence(1.0, 1.0) == pytest.approx(0.0)
    assert existence_confidence(20.0, 0.0) == pytest.approx(1.0, abs=1e-6)
    assert existence_confidence(0.0, 20.0) == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.parametrize(
    "question,expected",
    [("How many bikes are there?", True), ("What is the number of cars?", True), ("What color is the car?", False)],
)
def test_is_type2_question(question, expected):
    assert is_type2_question(question) is expected


def test_select_rois_type1_and_type2():
    ranked = [
        ROI((0, 0, 2, 2), 0.5, 0.9),
        ROI((1, 1, 3, 3), 0.4, 0.8),
        ROI((5, 5, 7, 7), 0.3, 0.7),
        ROI((8, 8, 9, 9), 0.9, 0.1),
    ]
    assert select_rois(ranked, type2=False, t_type2=0.5) == ranked[:1]
    merged = select_rois(ranked, type2=True, t_type2=0.5)
    assert [m.box for m in merged] == [(0, 0, 3, 3), (5, 5, 7, 7)]
    # Nothing above the threshold -> fall back to the most confident ROI.
    assert select_rois(ranked[3:], type2=True, t_type2=0.5) == ranked[3:]
    assert select_rois([], type2=True, t_type2=0.5) == []


def test_parse_targets():
    assert parse_targets(" I need the info about the soccer ball; water dispenser.\nQuestion: ...") == [
        "soccer ball",
        "water dispenser",
    ]
    assert parse_targets("car") == ["car"]
    assert parse_targets("I need the info about a; b; c; d; a", max_targets=3) == ["a", "b", "c"]
    assert parse_targets("   ") == []
    assert (
        parse_targets("I need the info about a long rambling answer that ignores the requested format entirely") == []
    )
    prompt = build_target_extraction_prompt("What is the color of the kite?")
    assert prompt.endswith("Question: What is the color of the kite?\nAnswer:")


class FakeBackend:
    """6x8 relevance grid over a 600x800 image; the 'kite' sits in grid cell (4, 6) = pixels (600..700, 400..500)."""

    GRID = (6, 8)
    OBJECT_PX = (650, 450)

    def __init__(self):
        self.calls = {"extract": 0, "values": 0, "exist": 0}

    def extract_targets(self, question):
        self.calls["extract"] += 1
        return "I need the info about kite."

    def value_features(self, image, target):
        self.calls["values"] += 1
        h, w = self.GRID
        n_visual = h * w
        values = 0.01 * torch.rand(2, n_visual + 3, 64, generator=torch.Generator().manual_seed(0))
        for i in range(n_visual + 3):
            values[:, i, i] += 1.0
        values[:, n_visual + 1] = values[:, 4 * w + 6]  # target token matches the object cell
        return ValueFeatures(values, [n_visual + 1], list(range(n_visual)), self.GRID)

    def existence_logits(self, images, target):
        self.calls["exist"] += len(images)
        return [(5.0, 0.0) if img.info.get("has_object") else (0.0, 5.0) for img in images]


def test_run_focus_end_to_end_with_fake_backend(monkeypatch):
    image = Image.new("RGB", (800, 600))
    # Tag crops that contain the object so the fake existence check can see it.
    original_crop = Image.Image.crop

    def crop(self, box=None):
        out = original_crop(self, box)
        x, y = FakeBackend.OBJECT_PX
        out.info["has_object"] = box[0] <= x < box[2] and box[1] <= y < box[3]
        return out

    monkeypatch.setattr(Image.Image, "crop", crop)
    backend = FakeBackend()
    config = FocusConfig(k=10, s_min=1, s_max=3, s_dist=2, n_steps=3, smooth_sigma=0.0)
    result = run_focus(backend, image, "What is the color of the kite?", config)

    assert result.targets == ["kite"] and not result.type2
    assert len(result.crops) == 1
    crop0 = result.crops[0]
    x, y = FakeBackend.OBJECT_PX
    assert crop0.box[0] <= x < crop0.box[2] and crop0.box[1] <= y < crop0.box[3]
    assert crop0.confidence > 0.9
    assert backend.calls == {"extract": 1, "values": 1, "exist": 3}
    assert result.num_forward_passes == 1 + 1 + 3

    # Known targets skip extraction.
    result = run_focus(backend, image, "What is the color of the kite?", config, targets=["kite"])
    assert backend.calls["extract"] == 1 and result.num_forward_passes == 4


def test_run_focus_skips_targets_without_tokens():
    class NoTokenBackend(FakeBackend):
        def value_features(self, image, target):
            feats = super().value_features(image, target)
            feats.target_idx = []
            return feats

    result = run_focus(NoTokenBackend(), Image.new("RGB", (800, 600)), "Where is the kite?", FocusConfig())
    assert result.targets == ["kite"] and result.crops == []
