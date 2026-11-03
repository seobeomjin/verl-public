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
"""CPU tests for the FOCUS-OPSD glue: teacher-logprob realignment and teacher prompt construction."""

from io import BytesIO

import pytest
import torch
from PIL import Image

from verl.experimental.agent_loop.focus_opsd_agent_loop import (
    _question_text,
    build_teacher_messages,
    load_original_image,
)
from verl.experimental.teacher_loop.teacher_manager import _align_teacher_to_student, _pad_teacher_outputs

PAD = -1


def _teacher_outputs(prompt_len, response, topk=1):
    """Fake vLLM prompt_logprobs: row i scores token i + 1 and the last row is a dummy."""
    n = prompt_len + len(response)
    ids = torch.zeros(n, topk, dtype=torch.int32)
    logprobs = torch.zeros(n, topk)
    for i in range(n - 1):
        ids[i] = 1000 + i  # prompt token ids are meaningless, only the response rows matter
        logprobs[i] = -float(i)
    for j, tok in enumerate(response):
        ids[prompt_len - 1 + j] = torch.arange(topk) * 100 + tok  # column 0 = scored token, then top-k
        logprobs[prompt_len - 1 + j] = -0.5 * (j + 1) - torch.arange(topk)
    return ids, logprobs


@pytest.mark.parametrize("teacher_prompt_len,student_prompt_len", [(9, 4), (4, 9), (5, 5), (1, 3)])
@pytest.mark.parametrize("topk", [1, 3])
def test_align_teacher_to_student_moves_response_rows(teacher_prompt_len, student_prompt_len, topk):
    response = [11, 12, 13]
    R = len(response)
    teacher_ids, teacher_logprobs = _teacher_outputs(teacher_prompt_len, response, topk)
    ids, logprobs = _align_teacher_to_student(
        teacher_ids, teacher_logprobs, teacher_prompt_len, student_prompt_len, R, pad_token_id=PAD
    )
    assert ids.shape == (student_prompt_len + R, topk) and logprobs.shape == ids.shape
    # The rows that score the response sit at the same offsets as in a plain prompt + response teacher call.
    plain_ids, plain_logprobs = _teacher_outputs(student_prompt_len, response, topk)
    rows = slice(student_prompt_len - 1, student_prompt_len - 1 + R)
    torch.testing.assert_close(ids[rows], plain_ids[rows])
    torch.testing.assert_close(logprobs[rows], plain_logprobs[rows])
    assert torch.all(ids[: student_prompt_len - 1] == PAD)
    assert torch.all(logprobs[: student_prompt_len - 1] == 0)
    # Downstream padding still works on the realigned layout.
    padded_ids, _ = _pad_teacher_outputs(ids, logprobs, 16, 8, student_prompt_len, R, PAD)
    assert padded_ids.shape == (1, 24, topk)


def test_align_teacher_to_student_rejects_length_mismatch():
    ids, logprobs = _teacher_outputs(5, [1, 2])
    with pytest.raises(AssertionError):
        _align_teacher_to_student(ids, logprobs, 6, 4, 2, PAD)


def _messages():
    image_item = {"type": "image", "image": "file:///data/a.jpg"}
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": [image_item, {"type": "text", "text": "What is the color of the kite?"}]},
    ]


def test_build_teacher_messages_inserts_crops_after_last_image_without_mutating_input():
    messages = _messages()
    crops = [("kite", Image.new("RGB", (4, 4))), ("tree", Image.new("RGB", (4, 4)))]
    teacher = build_teacher_messages(messages, crops, "Zoomed-in views:")
    content = teacher[1]["content"]
    assert [c["type"] for c in content] == ["image", "text", "text", "image", "text", "image", "text", "text"]
    assert content[1]["text"] == "\nZoomed-in views:" and content[2]["text"] == "\nkite:"
    assert content[6]["text"] == "\n"
    assert content[3]["image"] is crops[0][1] and content[5]["image"] is crops[1][1]
    assert content[-1]["text"] == "What is the color of the kite?"
    assert len(messages[1]["content"]) == 2  # student messages untouched
    assert build_teacher_messages([{"role": "user", "content": "no image"}], crops, "x") is None


def test_question_text():
    assert _question_text(_messages()) == "What is the color of the kite?"
    assert _question_text([{"role": "user", "content": "<image>Where is the dog?"}]) == "Where is the dog?"


def test_load_original_image(tmp_path):
    image = Image.new("RGB", (40, 30), color=(255, 0, 0))
    buf = BytesIO()
    image.save(buf, format="PNG")
    path = tmp_path / "a.png"
    image.save(path)
    assert load_original_image({"type": "image", "image": image}).size == (40, 30)
    assert load_original_image({"type": "image", "bytes": buf.getvalue()}).size == (40, 30)
    assert load_original_image({"type": "image", "image": {"bytes": buf.getvalue()}}).size == (40, 30)
    assert load_original_image({"type": "image", "image": f"file://{path}"}).size == (40, 30)
    assert load_original_image({"type": "image", "image": "https://example.com/a.png"}) is None
    assert load_original_image({"type": "image", "image": str(tmp_path / "missing.png")}) is None


class _FakeRemote:
    def __init__(self, fn):
        self.fn = fn

    async def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _FakeFocusWorker:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result, self.error = result, error
        self.focus = _FakeRemote(self._focus)

    def _focus(self, image, question, targets):
        self.calls.append((image.size, question, targets))
        if self.error:
            raise self.error
        return self.result


class _FakeBuilder:
    def build_initial_tokens(self, messages, images=None, **kwargs):
        self.messages, self.images = messages, images
        return list(range(100 + 10 * len(images)))


def _make_loop(monkeypatch, worker, validate=False, focus_enabled=True):
    import asyncio

    from verl.experimental.agent_loop import focus_opsd_agent_loop as mod
    from verl.experimental.agent_loop.agent_loop import AgentLoopOutput
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

    processed = Image.new("RGB", (400, 300))

    async def fake_run(self, sampling_params, priority=0, **kwargs):
        return AgentLoopOutput(
            prompt_ids=[1, 2, 3],
            response_ids=[4, 5],
            response_mask=[1, 1],
            multi_modal_data={"images": [processed]},
            num_turns=2,
            metrics={},
            extra_fields={},
        )

    monkeypatch.setattr(SingleTurnAgentLoop, "run", fake_run)
    import verl.experimental.focus.focus_worker as fw

    monkeypatch.setattr(fw, "get_or_create_focus_worker", lambda *a, **k: worker)
    loop = object.__new__(mod.FocusOPSDAgentLoop)
    loop.focus_enabled, loop.validate = focus_enabled, validate
    loop.focus_worker_name, loop.focus_num_gpus, loop.focus_worker_kwargs = "w", 0, {}
    loop.teacher_hint, loop.targets_key = "Zoom:", "focus_targets"
    loop.continuous_token_builder = _FakeBuilder()
    loop.loop = asyncio.get_event_loop()
    return loop


def _raw_prompt():
    original = Image.new("RGB", (800, 600), (0, 0, 255))  # 2x the rollout-side (processed) image
    return [
        {"role": "user", "content": [{"type": "image", "image": original}, {"type": "text", "text": "Kite color?"}]}
    ]


_RESULT = {
    "targets": ["kite"],
    "type2": False,
    "crops": [{"target": "kite", "box": (100, 50, 200, 150), "relevance": 0.1, "confidence": 0.9}],
    "num_forward_passes": 5,
}


@pytest.mark.asyncio
async def test_focus_loop_builds_teacher_context_from_full_resolution_crops(monkeypatch):
    worker = _FakeFocusWorker(_RESULT)
    loop = _make_loop(monkeypatch, worker)
    out = await loop.run({}, raw_prompt=_raw_prompt(), extra_info={"focus_targets": ["kite"]})
    assert worker.calls == [((400, 300), "Kite color?", ["kite"])]
    assert out.extra_fields["focus_num_crops"] == 1
    assert out.extra_fields["focus_detected_targets"] == ["kite"]
    images = out.extra_fields["teacher_multi_modal_data"]["images"]
    assert images[0].size == (400, 300)  # the image the student saw
    assert images[1].size == (200, 200)  # box scaled 2x onto the original
    assert out.extra_fields["teacher_prompt_ids"] == list(range(120))
    assert out.multi_modal_data["images"] == images[:1]  # student data untouched


@pytest.mark.asyncio
async def test_focus_loop_skips_validation_and_disabled_distillation(monkeypatch):
    for kwargs in ({"validate": True}, {"focus_enabled": False}):
        worker = _FakeFocusWorker(_RESULT)
        out = await _make_loop(monkeypatch, worker, **kwargs).run({}, raw_prompt=_raw_prompt())
        assert worker.calls == [] and "teacher_prompt_ids" not in out.extra_fields


@pytest.mark.asyncio
async def test_focus_loop_falls_back_without_crops_or_on_error(monkeypatch):
    for worker in (_FakeFocusWorker({**_RESULT, "crops": []}), _FakeFocusWorker(error=RuntimeError("boom"))):
        out = await _make_loop(monkeypatch, worker).run({}, raw_prompt=_raw_prompt())
        assert out.extra_fields["focus_num_crops"] == 0
        assert "teacher_prompt_ids" not in out.extra_fields and "teacher_multi_modal_data" not in out.extra_fields
