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
"""Frozen Hugging Face model that serves FOCUS (value-feature relevance maps + existence checks).

rollout engines (vLLM / SGLang) do not expose per-layer value features, so FOCUS runs on a separate
frozen HF copy of the model. Value features are captured with forward hooks on the language model's
``self_attn.v_norm`` (if present) or ``self_attn.v_proj``; values carry no RoPE, so sliding-window and
global attention layers are treated alike and no KV-cache layout is assumed.
"""

import hashlib
import logging
import os
import re
from collections import OrderedDict
from typing import Any, Optional

import torch
from PIL import Image

from verl.experimental.focus.pipeline import FocusConfig, ValueFeatures, run_focus
from verl.experimental.focus.prompts import (
    build_target_extraction_prompt,
    existence_question,
    relevance_question,
)
from verl.experimental.focus.relevance import find_target_token_span, infer_grid_hw
from verl.utils.tokenizer.tokenizer import get_processor_token_id

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_LAYER_PATTERN = re.compile(r"layers\.(\d+)\.self_attn\.(v_norm|v_proj)$")
_NON_LM_MARKERS = ("vision", "visual", "audio", "multi_modal_projector", "embed_vision")


def gemma4_grid_hw(image_processor, image_size: tuple[int, int]) -> tuple[int, int]:
    """Soft-token grid of a Gemma 4 image: aspect-preserving resize to multiples of ``patch * pooling``,
    then ``pooling x pooling`` average pooling laid out row-major (see ``Gemma4VisionPooler``)."""
    from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

    patch, pool = image_processor.patch_size, image_processor.pooling_kernel_size
    width, height = image_size
    target_h, target_w = get_aspect_ratio_preserving_size(
        height=height,
        width=width,
        patch_size=patch,
        max_patches=image_processor.max_soft_tokens * pool**2,
        pooling_kernel_size=pool,
    )
    return target_h // (patch * pool), target_w // (patch * pool)


class FocusWorker:
    """Runs the FOCUS pipeline with a frozen HF vision-language model.

    Args:
        model_path: HF model path (usually the student's initial checkpoint).
        focus_config: overrides for :class:`FocusConfig`.
        grid_mode: how to recover the visual-token grid: ``"auto"`` (``image_grid_thw`` if the processor
            returns it, else the aspect-ratio-matching factorization), ``"aspect"`` or ``"square"``.
        cache_size: number of results kept in the LRU cache (GRPO rollouts of one prompt share a result).
    """

    def __init__(
        self,
        model_path: str,
        focus_config: Optional[dict[str, Any]] = None,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
        trust_remote_code: bool = False,
        grid_mode: str = "auto",
        cache_size: int = 4096,
    ):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.config = FocusConfig.from_dict(focus_config)
        self.grid_mode = grid_mode
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        ).eval()
        self.image_token_id = get_processor_token_id(self.processor, "image")
        if self.image_token_id is None:
            raise ValueError(f"Could not resolve the image placeholder token of {model_path}")
        self.yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode("No", add_special_tokens=False)[0]

        self._captured: dict[int, torch.Tensor] = {}
        self._capturing = False
        self._register_value_hooks()
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_size = cache_size

    def _register_value_hooks(self):
        per_layer: dict[int, tuple[str, torch.nn.Module]] = {}
        for name, module in self.model.named_modules():
            match = _LAYER_PATTERN.search(name)
            if match is None or any(marker in name for marker in _NON_LM_MARKERS):
                continue
            layer, kind = int(match.group(1)), match.group(2)
            # Prefer the normalized value (what attention actually consumes) over the raw projection.
            if layer not in per_layer or kind == "v_norm":
                per_layer[layer] = (kind, module)
        if not per_layer:
            raise ValueError("No language-model value projections found to hook")
        layers = sorted(per_layer)
        start = int(len(layers) * self.config.layer_frac[0])
        end = max(start + 1, int(round(len(layers) * self.config.layer_frac[1])))
        self.selected_layers = layers[start:end]
        logger.info(f"FOCUS uses value features of layers {self.selected_layers} ({len(layers)} total)")

        def make_hook(layer: int):
            def hook(module, inputs, output):
                if self._capturing:
                    self._captured[layer] = output.detach()

            return hook

        for layer in self.selected_layers:
            per_layer[layer][1].register_forward_hook(make_hook(layer))

    def _encode(self, text: str, images: Optional[list[Image.Image]]) -> dict[str, torch.Tensor]:
        content = [{"type": "image"} for _ in images or []] + [{"type": "text", "text": text}]
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=[prompt], images=images or None, return_tensors="pt")
        return inputs.to(self.model.device)

    def _collect_values(self, seq_len: int) -> list[torch.Tensor]:
        """Per-layer ``[N, D_l]`` value features; widths may differ (e.g. Gemma 4 sliding vs. global layers)."""
        device = self._captured[next(iter(self._captured))].device
        values = []
        for layer in self.selected_layers:
            if layer not in self._captured:
                continue
            v = self._captured[layer][0]
            if v.dim() == 3 and v.shape[0] != seq_len and v.shape[1] == seq_len:  # [heads, N, D]
                v = v.transpose(0, 1)
            values.append(v.reshape(seq_len, -1).to(device))
        return values

    def _grid_hw(self, inputs: dict[str, torch.Tensor], num_visual: int, image: Image.Image) -> tuple[int, int]:
        image_processor = self.processor.image_processor
        if self.grid_mode == "auto" and "image_grid_thw" in inputs:  # Qwen-VL family
            _, h, w = inputs["image_grid_thw"][0].tolist()
            merge = getattr(image_processor, "merge_size", 1)
            return h // merge, w // merge
        if self.grid_mode == "auto" and hasattr(image_processor, "pooling_kernel_size"):  # Gemma 4
            grid = gemma4_grid_hw(image_processor, image.size)
            if grid[0] * grid[1] == num_visual:
                return grid
        if self.grid_mode == "square":
            side = int(round(num_visual**0.5))
            if side * side == num_visual:
                return side, side
        return infer_grid_hw(num_visual, *image.size)

    @torch.no_grad()
    def extract_targets(self, question: str) -> str:
        inputs = self._encode(build_target_extraction_prompt(question), None)
        out = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return self.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    @torch.no_grad()
    def value_features(self, image: Image.Image, target: str) -> Optional[ValueFeatures]:
        inputs = self._encode(relevance_question(target), [image])
        self._captured.clear()
        self._capturing = True
        try:
            self.model(**inputs, use_cache=False)
        finally:
            self._capturing = False
        input_ids = inputs["input_ids"][0]
        visual_idx = (input_ids == self.image_token_id).nonzero().flatten().tolist()
        if not visual_idx or not self._captured:
            return None
        token_strs = [self.tokenizer.decode([t]) for t in input_ids.tolist()]
        target_idx = find_target_token_span(token_strs, target, search_start=visual_idx[-1] + 1)
        return ValueFeatures(
            values=self._collect_values(len(input_ids)),
            target_idx=target_idx,
            visual_idx=visual_idx,
            grid_hw=self._grid_hw(inputs, len(visual_idx), image),
        )

    @torch.no_grad()
    def existence_logits(self, images: list[Image.Image], target: str) -> list[tuple[float, float]]:
        results = []
        for image in images:
            logits = self.model(**self._encode(existence_question(target), [image]), use_cache=False).logits[0, -1]
            results.append((logits[self.yes_id].item(), logits[self.no_id].item()))
        return results

    def focus(self, image: Image.Image, question: str, targets: Optional[list[str]] = None) -> dict[str, Any]:
        """Run FOCUS and return :class:`FocusResult` as a dict (boxes in ``image`` pixel coordinates)."""
        key = hashlib.sha1(
            image.tobytes() + repr((image.size, question, targets)).encode(), usedforsecurity=False
        ).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        result = run_focus(self, image, question, self.config, targets).to_dict()
        self._cache[key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result


def get_or_create_focus_worker(name: str, num_gpus: float, **worker_kwargs):
    """Get the named FOCUS Ray actor, creating it on first use (shared by all agent loop workers)."""
    import ray

    return (
        ray.remote(FocusWorker)
        .options(name=name, get_if_exists=True, num_gpus=num_gpus, max_concurrency=1)
        .remote(**worker_kwargs)
    )
