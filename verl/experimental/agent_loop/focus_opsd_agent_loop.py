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
"""On-policy self-distillation whose teacher additionally sees FOCUS crops.

The student rolls out on the original image only. FOCUS (arXiv:2506.21710) localizes the question's
target object(s) with a frozen copy of the model and the resulting zoomed-in crops are appended to the
teacher's prompt, so the teacher scores the student's response with privileged fine-grained detail.
Configure the loop through ``actor_rollout_ref.rollout.agent.agent_loop_config_path``, e.g.::

    - name: focus_opsd_agent
      _target_: verl.experimental.agent_loop.focus_opsd_agent_loop.FocusOPSDAgentLoop
      focus_num_gpus: 2
      focus_config: {k: 30, s_max: 5, n_steps: 4}
"""

import logging
import os
import time
from io import BytesIO
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.experimental.focus.prompts import DEFAULT_TEACHER_HINT
from verl.experimental.focus.roi import scale_box
from verl.trainer.distillation import is_distillation_enabled
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def load_original_image(item: dict[str, Any]) -> Optional[Image.Image]:
    """Decode a dataset image item without the rollout-side resizing; ``None`` if it cannot be loaded locally."""
    image = item.get("image", item)
    try:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict) and image.get("bytes") is not None:
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
        if isinstance(image, str) and not image.startswith(("http://", "https://", "data:")):
            return Image.open(image.removeprefix("file://")).convert("RGB")
    except (OSError, ValueError) as e:
        logger.warning(f"Could not load the original image for FOCUS crops: {e}")
    return None


def _to_container(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    return dict(cfg)


def _find_last_image(messages: list[dict[str, Any]]) -> Optional[tuple[int, int]]:
    position = None
    for i, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            for j, item in enumerate(content):
                if isinstance(item, dict) and item.get("type") == "image":
                    position = (i, j)
    return position


def _question_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.replace("<image>", "").strip()
        return "\n".join(
            item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return ""


def build_teacher_messages(
    messages: list[dict[str, Any]], crops: list[tuple[str, Image.Image]], hint: str
) -> Optional[list[dict[str, Any]]]:
    """Insert ``hint`` and the labeled crops right after the last image, so the crops come last in image order."""
    position = _find_last_image(messages)
    if position is None:
        return None
    msg_idx, item_idx = position
    # Chat templates such as Gemma's concatenate content items verbatim, so separators are explicit.
    extra: list[dict[str, Any]] = [{"type": "text", "text": f"\n{hint}"}]
    for target, crop in crops:
        extra += [{"type": "text", "text": f"\n{target}:"}, {"type": "image", "image": crop}]
    extra.append({"type": "text", "text": "\n"})
    teacher_messages = [dict(m) for m in messages]
    content = list(teacher_messages[msg_idx]["content"])
    teacher_messages[msg_idx]["content"] = content[: item_idx + 1] + extra + content[item_idx + 1 :]
    return teacher_messages


@register("focus_opsd_agent")
class FocusOPSDAgentLoop(SingleTurnAgentLoop):
    """Single-turn rollout whose distillation teacher also sees FOCUS crops of the input image.

    Args:
        focus_model_path: frozen model used by FOCUS; defaults to the student's initial checkpoint.
        focus_worker_name: name of the shared FOCUS Ray actor.
        focus_num_gpus: GPUs reserved for the FOCUS actor (must be free outside the trainer's pools).
        focus_config: overrides for :class:`verl.experimental.focus.pipeline.FocusConfig`.
        focus_worker_kwargs: extra kwargs for :class:`verl.experimental.focus.focus_worker.FocusWorker`.
        teacher_hint: text placed before the crops in the teacher prompt.
        targets_key: optional per-sample field (top level or in ``extra_info``) with known target objects,
            which skips FOCUS's target extraction step.
    """

    def __init__(
        self,
        *args,
        focus_model_path: Optional[str] = None,
        focus_worker_name: str = "focus_worker",
        focus_num_gpus: float = 1,
        focus_config: Optional[dict[str, Any]] = None,
        focus_worker_kwargs: Optional[dict[str, Any]] = None,
        teacher_hint: str = DEFAULT_TEACHER_HINT,
        targets_key: str = "focus_targets",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.focus_enabled = is_distillation_enabled(self.config.get("distillation"))
        self.focus_worker_name = focus_worker_name
        self.focus_num_gpus = focus_num_gpus
        # Hydra passes yaml kwargs as DictConfig; hand plain containers to the Ray actor.
        self.focus_worker_kwargs = dict(
            model_path=focus_model_path or self.config.actor_rollout_ref.model.path,
            focus_config=_to_container(focus_config),
            trust_remote_code=self.config.actor_rollout_ref.model.get("trust_remote_code", False),
            **_to_container(focus_worker_kwargs),
        )
        self.teacher_hint = teacher_hint
        self.targets_key = targets_key

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        output = await super().run(sampling_params, priority=priority, **kwargs)
        # The teacher only scores training rollouts, so FOCUS is skipped for validation.
        if not self.focus_enabled or self.validate:
            return output

        start = time.perf_counter()
        try:
            teacher_context = await self._build_teacher_context(list(kwargs["raw_prompt"]), output, kwargs)
        except Exception as e:
            logger.warning(f"FOCUS failed, falling back to the plain teacher prompt: {e}")
            teacher_context = None
        output.extra_fields["focus_time"] = time.perf_counter() - start
        output.extra_fields["focus_num_crops"] = len(teacher_context["crops"]) if teacher_context else 0
        if teacher_context:
            output.extra_fields["focus_detected_targets"] = teacher_context["targets"]
            output.extra_fields["teacher_prompt_ids"] = teacher_context["prompt_ids"]
            output.extra_fields["teacher_multi_modal_data"] = {"images": teacher_context["images"]}
        return output

    def _known_targets(self, kwargs: dict[str, Any]) -> Optional[list[str]]:
        targets = kwargs.get(self.targets_key)
        if targets is None:
            targets = (kwargs.get("extra_info") or {}).get(self.targets_key)
        if targets is None:
            return None
        targets = [targets] if isinstance(targets, str) else [str(t) for t in targets]
        return targets or None

    async def _build_teacher_context(
        self, messages: list[dict[str, Any]], output: AgentLoopOutput, kwargs: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        from verl.experimental.focus.focus_worker import get_or_create_focus_worker

        images = (output.multi_modal_data or {}).get("images")
        position = _find_last_image(messages)
        if not images or position is None:
            return None
        # FOCUS localizes on the image the student saw; crops are cut from the full-resolution original.
        image = images[0]
        original = load_original_image(messages[position[0]]["content"][position[1]]) if len(images) == 1 else None
        original = original or image

        worker = get_or_create_focus_worker(self.focus_worker_name, self.focus_num_gpus, **self.focus_worker_kwargs)
        result = await worker.focus.remote(image, _question_text(messages), self._known_targets(kwargs))
        if not result["crops"]:
            return None

        crops = [
            (c["target"], original.crop(scale_box(tuple(c["box"]), image.size, original.size))) for c in result["crops"]
        ]
        teacher_messages = build_teacher_messages(messages, crops, self.teacher_hint)
        if teacher_messages is None:
            return None
        teacher_images = list(images) + [crop for _, crop in crops]
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.continuous_token_builder.build_initial_tokens(teacher_messages, images=teacher_images),
        )
        return {"prompt_ids": prompt_ids, "images": teacher_images, "crops": crops, "targets": result["targets"]}
