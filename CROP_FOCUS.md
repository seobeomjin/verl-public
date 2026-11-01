# FOCUS crops as privileged teacher context for on-policy self-distillation

This branch (`feature/crop-focus`) adds [FOCUS](https://arxiv.org/abs/2506.21710) (training-free visual cropping)
to verl's on-policy distillation (OPSD) path. Status: **CPU-tested only**; GPU and end-to-end runs are not done yet.

## Purpose

Fine details in high-resolution images are often lost at the model's input resolution. FOCUS finds the image
regions relevant to a question and crops them. Used at inference, crops help but cost extra forward passes.

Here the crops are used at **training time only**, as privileged information:

- The **student** (being trained) rolls out on the **original image only**.
- The **teacher** (a frozen copy of the same checkpoint) scores the student's response with the **original image
  plus the FOCUS crops** in its prompt.
- The existing distillation loss pulls the student toward "what the model would say if it could see the crops",
  so the student can internalize fine-grained perception and does not need crops at inference.

Target model: `gemma-4-31B-it-max560` (`max_soft_tokens=560`).

## How FOCUS works (paper, summarized)

1. **Target extraction**: a few-shot prompt asks which object(s) the question is about.
2. **V-V pseudo-attention**: prefill `"Is there a {target} in the image?"` with the image. For each selected layer,
   take the cosine similarity between the target token's *value* vector and every token's value vector. Value
   vectors are used instead of Q/K, so no attention matrix is needed and flash attention stays usable.
3. **Aggregate**: apply `(A + I) / 2`, row-normalize, sum over layers (attention-rollout style), multiply across
   target tokens (consensus), optional Gaussian smoothing -> an `h x w` relevance map over the image-token grid.
4. **Propose ROIs**: top-k anchors (min distance `s_dist`), grow each box from `s_min` to `s_max` while it stays
   relevant, then NMS.
5. **Re-rank**: ask `"Is there a {target} in the image? Answer Yes or No."` on each candidate crop;
   `c_yes = 2 * (softmax([l_yes, l_no])_yes - 0.5)`.
6. **Select**: type-1 questions take the top ROI per target; type-2 (counting) take every ROI above `t_type2`,
   with overlapping ROIs merged.

## Data flow

```
AgentLoopWorker
  -> FocusOPSDAgentLoop.run()
       1. SingleTurnAgentLoop.run()            student generates on the original image (unchanged path)
       2. FocusWorker.focus(image, question)   Ray actor, frozen HF model, steps I-V
       3. crops cut from the ORIGINAL-resolution image (the rollout image may be downscaled)
       4. teacher prompt = original image + labeled crops + question, tokenized with the continuous token builder
       -> extra_fields: teacher_prompt_ids, teacher_multi_modal_data
  -> _agent_loop_postprocess pops them (they are never sent to the reward worker)
  -> _compute_teacher_logprobs scores teacher_prompt_ids + response_ids with the teacher vLLM server
  -> _align_teacher_to_student moves the response rows onto the student's prompt length
  -> existing _pad_teacher_outputs / distillation loss
```

Failure modes (no target, no ROI, any exception) fall back to the plain path with no crops. FOCUS is skipped during
validation and when distillation is disabled.

## Code changes

New:

| Path | Role |
| --- | --- |
| `verl/experimental/focus/relevance.py` | Pure-torch relevance map (Fig. 7 of the paper), Gaussian smoothing, grid inference, target-token span lookup |
| `verl/experimental/focus/roi.py` | Anchors, ROI expansion, NMS, coordinate mapping, existence confidence, type-1/2 selection |
| `verl/experimental/focus/prompts.py` | Target-extraction few-shot prompt and parser, existence prompt, teacher hint |
| `verl/experimental/focus/pipeline.py` | `FocusConfig`, `FocusBackend` protocol, `run_focus` (model-agnostic orchestration) |
| `verl/experimental/focus/focus_worker.py` | `FocusWorker` Ray actor: HF model, value hooks, exact visual-token grid, LRU cache |
| `verl/experimental/agent_loop/focus_opsd_agent_loop.py` | `focus_opsd_agent` loop that builds the crop-augmented teacher context |
| `examples/on_policy_distillation_trainer/run_gemma4_31b_focus_opsd_fsdp.sh`, `config/focus_opsd_agent_loop.yaml` | Example run and FOCUS hyperparameters |
| `scripts/eval_focus_roi.py` | ROI-recall evaluation against ground-truth boxes (e.g. V*Bench) |
| `tests/experimental/focus/` | CPU tests |

Modified (small, additive; behaviour is unchanged when no teacher context is present):

- `verl/experimental/agent_loop/agent_loop.py`: `AgentLoopBase.validate` flag; module-level `pop_teacher_context`;
  `_compute_teacher_logprobs` accepts `teacher_prompt_ids` / `teacher_multi_modal_data`.
- `verl/experimental/teacher_loop/teacher_manager.py`: `_align_teacher_to_student`. vLLM `prompt_logprobs` row `i`
  scores token `i+1`, so the teacher's rows `[Tp-1 : Tp+R]` are placed at the student's `[P-1 : P+R]` and the front is
  padded. Those positions are masked by `response_mask` in the loss.
- `verl/trainer/ppo/v1/agent_loop_tq.py`: the same wiring for the TQ path.
- `verl/experimental/agent_loop/__init__.py`: registers the new loop.

## Design notes and deviations from the paper

- **Value features come from a separate frozen HF model**, not the vLLM engine, because vLLM does not expose
  value vectors. This costs extra GPUs (the example needs 14: trainer 8, teacher 4, FOCUS 2).
- **Steps I and V run in `FocusWorker`**, not on the teacher vLLM server, because an agent loop has no teacher client.
- **FOCUS runs after student generation**, not in parallel, to avoid processing the image twice. Results are cached,
  so the n rollouts of one prompt share one FOCUS run.
- **Gemma4-31B specifics** (checked against the real config and processor):
  - 60 layers; every 6th is global attention with `attention_k_eq_v=True`, so there is no `v_proj`. The hook is
    placed on `v_norm`, which exists on every layer.
  - Value widths differ between sliding (16x256) and global (4x512) layers, so features are kept as a per-layer list.
  - The image-token grid is computed exactly from the processor (it matches the soft-token count), not approximated.
- **Gaussian smoothing uses normalized convolution** (`blur(x) / blur(ones)`). Reflect padding moved the peak toward
  the border on small maps.
- Dataset rows may carry `focus_targets` (in `extra_info`) to skip target extraction.

## Tests

```bash
pytest tests/experimental/focus -q          # 43 CPU tests
```

Covered: equivalence to the paper's reference implementation, ROI logic, pipeline with a fake backend, logprob
realignment (teacher prompt longer, shorter, or equal), teacher message construction, the agent loop with a fake
worker (crop scaling, validation skip, fallback). `FocusWorker` was also run end to end on a tiny random Gemma4 with
the 31B layout and the real 31B processor (fp32 and bf16). Existing agent-loop, TQ and distillation CPU tests still
pass.

## Not verified yet

- Real-weight behaviour: ROI recall on V*Bench (`scripts/eval_focus_roi.py`) and accuracy with crops.
- A short end-to-end OPSD run on GPUs.
- Training dataset choice (V*/HRBench are evaluation-only) and `s_max` for 4K/8K images (the relevance map is only
  about 19x28 cells for Gemma4-560).

## Relation to upstream PRs

Open PRs #6833, #6909 and #7659 add text-only privileged teacher context with the same "score a longer teacher prompt,
then realign to the student" idea. None of them handles images or crops. `_align_teacher_to_student` keeps the same
semantics as #6833's `slice_privileged_teacher_to_student`; if #6833 merges first, switch to it.
