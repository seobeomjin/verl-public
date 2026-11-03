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
"""Object relevance maps from cached value features (FOCUS, arXiv:2506.21710, Sec. 3.2 / Fig. 7).

For each target text token, the value feature is compared (cosine similarity) against the value
features of all tokens in a layer. Layers are aggregated rollout-style with a residual identity
term, and target tokens are fused by element-wise product so only regions consistently highlighted
by every target token stay prominent. Everything here is model-agnostic and CPU-testable.
"""

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def compute_relevance_map(
    values: torch.Tensor | Sequence[torch.Tensor],
    target_idx: Sequence[int],
    visual_idx: Sequence[int] | torch.Tensor,
    grid_hw: tuple[int, int],
    smooth_sigma: float = 0.0,
) -> torch.Tensor:
    """Compute the object relevance map of a target object.

    Args:
        values: value features of the selected layers, ``[L, N, D]`` or a list of ``[N, D_l]`` tensors
            (heads flattened into D). A list allows layers with different value widths, e.g. Gemma 4's
            sliding vs. global attention layers.
        target_idx: sequence positions of the target object's text tokens.
        visual_idx: sequence positions of the visual tokens, in row-major grid order.
        grid_hw: ``(h, w)`` with ``h * w == len(visual_idx)``.
        smooth_sigma: std of the Gaussian filter applied to the map (in grid cells); 0 disables it.

    Returns:
        ``[h, w]`` float32 map, non-negative and summing to 1.
    """
    if len(target_idx) == 0:
        raise ValueError("target_idx must not be empty")
    device = values[0].device
    visual_idx = torch.as_tensor(visual_idx, dtype=torch.long, device=device)
    h, w = grid_hw
    if h * w != visual_idx.numel():
        raise ValueError(f"grid {grid_hw} does not match {visual_idx.numel()} visual tokens")

    layers = [F.normalize(v.float().to(device), dim=-1) for v in values]  # each N, D_l
    num_tokens = layers[0].shape[0]
    rel_map = torch.ones(visual_idx.numel(), dtype=torch.float32, device=device)
    for t in target_idx:
        # Row ``t`` of each layer's cosine-similarity matrix: L, N
        row = torch.stack([x @ x[t] for x in layers])
        # Residual connection ((A + I) / 2) followed by row-wise normalization, as in attention rollout.
        identity = F.one_hot(torch.tensor(t, device=device), num_tokens).to(row.dtype)
        row = (row + identity) / 2
        row = row / row.sum(dim=-1, keepdim=True)
        pseudo_attn = row[:, visual_idx].sum(dim=0)
        # Cosine similarities can be negative; clamp before the product so signs cannot flip the consensus.
        pseudo_attn = pseudo_attn.clamp_min(0)
        pseudo_attn = pseudo_attn / pseudo_attn.sum().clamp_min(1e-12)
        rel_map = rel_map * pseudo_attn
        rel_map = rel_map / rel_map.sum().clamp_min(1e-12)

    rel_map = rel_map.reshape(h, w)
    if smooth_sigma > 0:
        rel_map = gaussian_blur(rel_map, smooth_sigma)
        rel_map = rel_map / rel_map.sum().clamp_min(1e-12)
    return rel_map


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a 2D map with a separable Gaussian kernel.

    Uses normalized convolution (zero-padded blur divided by the blurred all-ones mask) so cells near
    the border are averaged only over in-map neighbors; reflect/replicate padding would bias small maps
    towards their edges.
    """
    radius = max(1, int(math.ceil(3 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    def blur(t: torch.Tensor) -> torch.Tensor:
        t = F.conv2d(t[None, None], kernel.view(1, 1, 1, -1), padding=(0, radius))
        return F.conv2d(t, kernel.view(1, 1, -1, 1), padding=(radius, 0))[0, 0]

    return blur(x) / blur(torch.ones_like(x))


def infer_grid_hw(num_tokens: int, image_width: int, image_height: int) -> tuple[int, int]:
    """Pick the ``(h, w)`` factorization of ``num_tokens`` whose aspect ratio best matches the image.

    Used for processors that lay visual tokens out as an aspect-preserving 2D grid without exposing
    the grid shape (e.g. Gemma 4's variable-resolution soft tokens).
    """
    target = math.log(image_width / image_height)
    best = None
    for h in range(1, num_tokens + 1):
        if num_tokens % h:
            continue
        w = num_tokens // h
        err = abs(math.log(w / h) - target)
        if best is None or err < best[0]:
            best = (err, h, w)
    return best[1], best[2]


def find_target_token_span(token_strs: Sequence[str], target: str, search_start: int = 0) -> list[int]:
    """Return the indices of the tokens covering the last occurrence of ``target``.

    Works on per-token decoded strings, so it is independent of how the tokenizer splits words
    (leading spaces, sub-words, ...). Matching is case-insensitive. Returns ``[]`` if not found.
    """
    pieces = list(token_strs[search_start:])
    starts, pos = [], 0
    for piece in pieces:
        starts.append(pos)
        pos += len(piece)
    text = "".join(pieces).lower()
    char_start = text.rfind(target.strip().lower())
    if char_start < 0 or not target.strip():
        return []
    char_end = char_start + len(target.strip())
    return [
        search_start + i
        for i, (s, piece) in enumerate(zip(starts, pieces, strict=True))
        if s < char_end and s + len(piece) > char_start and piece.strip()
    ]
