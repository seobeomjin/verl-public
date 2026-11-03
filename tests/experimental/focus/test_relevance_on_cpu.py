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
"""CPU tests for FOCUS object relevance maps."""

import pytest
import torch

from verl.experimental.focus.relevance import (
    compute_relevance_map,
    find_target_token_span,
    gaussian_blur,
    infer_grid_hw,
)

GRID = (4, 4)
NUM_PREFIX = 2  # text tokens before the image


def _make_values(num_layers=3, dim=64, seed=0):
    """Sequence = 2 prefix tokens, 16 visual tokens (4x4), 3 trailing text tokens; one-hot-ish features."""
    g = torch.Generator().manual_seed(seed)
    n = NUM_PREFIX + GRID[0] * GRID[1] + 3
    values = 0.05 * torch.rand(num_layers, n, dim, generator=g)
    for i in range(n):  # mostly orthogonal tokens
        values[:, i, i % dim] += 1.0
    visual_idx = list(range(NUM_PREFIX, NUM_PREFIX + GRID[0] * GRID[1]))
    return values, visual_idx


def _cell(y, x):
    return NUM_PREFIX + y * GRID[1] + x


def _reference_fig7(values, target_idx, visual_idx):
    """Direct transcription of the paper's Fig. 7 pseudocode (full N x N similarity)."""
    x = torch.nn.functional.normalize(values.float(), dim=-1)
    sim = torch.bmm(x, x.transpose(1, 2))  # L, N, N
    rel = torch.ones(len(visual_idx))
    for t in target_idx:
        pseudo = torch.zeros(len(visual_idx))
        for layer in range(values.shape[0]):
            m = (sim[layer] + torch.eye(sim.shape[1])) / 2
            m = m / m.sum(dim=-1, keepdim=True)
            pseudo += m[t, visual_idx]
        pseudo /= pseudo.sum()
        rel *= pseudo
    return rel / rel.sum()


def test_matches_paper_pseudocode_for_non_negative_features():
    values, visual_idx = _make_values()
    target_idx = [NUM_PREFIX + 16, NUM_PREFIX + 17]
    ours = compute_relevance_map(values, target_idx, visual_idx, GRID).flatten()
    torch.testing.assert_close(ours, _reference_fig7(values, target_idx, visual_idx))


def test_peak_at_matching_visual_token():
    values, visual_idx = _make_values()
    target = NUM_PREFIX + 16
    values[:, target] = values[:, _cell(1, 2)]
    rel = compute_relevance_map(values, [target], visual_idx, GRID)
    assert rel.shape == GRID
    assert torch.argmax(rel).item() == 1 * GRID[1] + 2
    assert torch.all(rel >= 0) and torch.isclose(rel.sum(), torch.tensor(1.0))


def test_target_tokens_fuse_by_consensus():
    # "red" matches cells A and B, "car" matches B and C -> only B survives the product.
    values, visual_idx = _make_values()
    red, car = NUM_PREFIX + 16, NUM_PREFIX + 17
    a, b, c = _cell(0, 0), _cell(2, 2), _cell(3, 1)
    values[:, red] = values[:, a] + values[:, b]
    values[:, car] = values[:, b] + values[:, c]
    single = compute_relevance_map(values, [red], visual_idx, GRID).flatten()
    fused = compute_relevance_map(values, [red, car], visual_idx, GRID).flatten()
    assert set(torch.topk(single, 2).indices.tolist()) == {0, 10}
    assert torch.argmax(fused).item() == 2 * GRID[1] + 2
    assert fused[2 * GRID[1] + 2] > 2 * fused[0]


def test_negative_similarities_do_not_flip_consensus():
    values, visual_idx = _make_values()
    t1, t2 = NUM_PREFIX + 16, NUM_PREFIX + 17
    values[:, t1] = values[:, _cell(1, 1)]
    values[:, t2] = values[:, _cell(1, 1)]
    values[:, _cell(3, 3)] = -values[:, t1]  # anti-correlated with both targets
    rel = compute_relevance_map(values, [t1, t2], visual_idx, GRID)
    assert torch.argmax(rel).item() == 1 * GRID[1] + 1
    assert rel[3, 3] == 0


def test_smoothing_keeps_peak_and_normalization():
    values, visual_idx = _make_values()
    target = NUM_PREFIX + 16
    values[:, target] = values[:, _cell(2, 1)]
    rel = compute_relevance_map(values, [target], visual_idx, GRID, smooth_sigma=1.0)
    assert torch.argmax(rel).item() == 2 * GRID[1] + 1
    assert torch.isclose(rel.sum(), torch.tensor(1.0))


def test_gaussian_blur_preserves_mass_of_constant_map():
    x = torch.full((5, 7), 2.0)
    torch.testing.assert_close(gaussian_blur(x, 1.0), x)
    # Degenerate 1-row map must not crash on reflect padding.
    assert gaussian_blur(torch.rand(1, 6), 1.0).shape == (1, 6)


def test_invalid_inputs():
    values, visual_idx = _make_values()
    with pytest.raises(ValueError):
        compute_relevance_map(values, [], visual_idx, GRID)
    with pytest.raises(ValueError):
        compute_relevance_map(values, [NUM_PREFIX + 16], visual_idx, (3, 4))


@pytest.mark.parametrize(
    "num_tokens,size,expected",
    [(532, (2246, 1582), (19, 28)), (517, (500, 2000), (47, 11)), (16, (100, 100), (4, 4)), (7, (10, 10), (1, 7))],
)
def test_infer_grid_hw(num_tokens, size, expected):
    assert infer_grid_hw(num_tokens, *size) == expected


def test_find_target_token_span_handles_subwords_and_case():
    tokens = ["<img>", "<img>", "Is", " there", " a", " Par", "ag", "lider", " in", " the", " image", "?"]
    assert find_target_token_span(tokens, "paraglider", search_start=2) == [5, 6, 7]
    assert find_target_token_span(tokens, "the image") == [9, 10]
    assert find_target_token_span(tokens, "balloon") == []
    assert find_target_token_span(tokens, "  ") == []


def test_find_target_token_span_uses_last_occurrence_after_start():
    tokens = ["car", " and", " the", " car", "?"]
    assert find_target_token_span(tokens, "car") == [3]
    assert find_target_token_span(tokens, "car", search_start=4) == []


def test_layers_with_different_value_widths():
    # Gemma 4 mixes sliding (16 x 256) and global (4 x 512) value widths across layers.
    values, visual_idx = _make_values(num_layers=2, dim=64)
    target = NUM_PREFIX + 16
    values[:, target] = values[:, _cell(3, 0)]
    narrow = values[1, :, :32].clone()
    rel = compute_relevance_map([values[0], narrow], [target], visual_idx, GRID)
    assert torch.argmax(rel).item() == 3 * GRID[1] + 0
    # A list of equal-width layers is equivalent to the stacked tensor.
    torch.testing.assert_close(
        compute_relevance_map(list(values), [target], visual_idx, GRID),
        compute_relevance_map(values, [target], visual_idx, GRID),
    )
