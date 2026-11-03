#!/usr/bin/env bash
# On-policy self-distillation with FOCUS crops | vision (Gemma 4) | vLLM rollout | FSDP training | NVIDIA GPUs
#
# The student rolls out on the original image. A frozen copy of the same checkpoint is the teacher; its prompt
# additionally contains zoomed-in crops of the question's target object(s) found by FOCUS (arXiv:2506.21710),
# so it scores the student's response with privileged fine-grained detail.
#
# GPU layout (defaults): the Ray cluster needs >= 14 GPUs: trainer 8 | teacher vLLM 4 | FOCUS HF actor 2.
# The FOCUS actor (focus_num_gpus in the agent loop config) takes GPUs outside the trainer / teacher pools.

set -xeuo pipefail

# ---- user-adjustable ----
MODEL=${MODEL:-google/gemma-4-31B-it}  # e.g. a local gemma-4-31B-it-max560 checkpoint
TEACHER_MODEL=${TEACHER_MODEL:-$MODEL}  # self-distillation: teacher = frozen student init

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
rollout_n=${ROLLOUT_N:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
# Teacher prompts carry up to max_crops extra images (<= 560 soft tokens each for max560 checkpoints).
teacher_extra_tokens=${TEACHER_EXTRA_TOKENS:-2560}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}
teacher_tp=${TEACHER_TP:-4}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.8}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-100}
test_freq=${TEST_FREQ:-10}

project_name=${PROJECT_NAME:-verl_focus_opsd}
experiment_name=${EXPERIMENT_NAME:-gemma4_31b_focus_opsd_vllm_fsdp}
# ---- end user-adjustable ----

train_file=${TRAIN_FILE:-$HOME/data/hr_vqa/train.parquet}
val_file=${VAL_FILE:-$HOME/data/hr_vqa/test.parquet}
agent_loop_config=${AGENT_LOOP_CONFIG:-examples/on_policy_distillation_trainer/config/focus_opsd_agent_loop.yaml}

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
teacher_max_num_tokens=$(( max_num_tokens + teacher_extra_tokens ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$train_file']"
    data.val_files="['$val_file']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.image_key=images
)

MODEL_ARGS=(
    actor_rollout_ref.model.path="$MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config}
    actor_rollout_ref.rollout.agent.default_agent_loop=focus_opsd_agent
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm × fsdp extras of the committed uv.lock;
# NPU falls back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
