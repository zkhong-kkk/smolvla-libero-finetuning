# SmolVLA Fine-Tuning and Evaluation on LIBERO

This repository documents a single-GPU SmolVLA fine-tuning project on the
LIBERO benchmark. The project covers model fine-tuning, checkpoint validation,
simulation evaluation, and a controlled comparison with the released SmolVLA
LIBERO checkpoint.

The experiments were run on one NVIDIA RTX 4090 using LeRobot 0.6.2. The main
goal is to study whether continued fine-tuning improves manipulation success
and to build a reproducible evaluation pipeline that can later be extended to
robustness testing.

## Technical Reports

- [中文技术报告](docs/TECHNICAL_REPORT_ZH.md)
- [English Technical Report](docs/TECHNICAL_REPORT_EN.md)
- [Evaluation Guide / 评测指南](docs/EVALUATION_GUIDE.md)
- [Training Configuration / 训练配置](docs/TRAINING_CONFIGURATION.md)
- [Hugging Face Model Card](docs/HUGGINGFACE_MODEL_CARD.md)

## Released Checkpoint

The final Stage 2 policy is published on Hugging Face:

- [zkhomh-kkk/smolvla-libero-spatial-stage2](https://huggingface.co/zkhomh-kkk/smolvla-libero-spatial-stage2)

Use the Hub repository ID anywhere LeRobot accepts a policy path:

```bash
MODEL_PATH="zkhomh-kkk/smolvla-libero-spatial-stage2"
```

## What This Project Adds

The model architecture, released weights, LeRobot framework, and LIBERO tasks
come from their respective upstream projects. This repository adds a
single-RTX-4090 two-stage continuation study, a controlled comparison between
local 20k/50k checkpoints and the released checkpoint, camera-interface
adaptation for closed-loop LIBERO rollouts, and an exploratory visual
corruption and uncertainty analysis. The contribution is experimental and
engineering-focused; it does not claim a new VLA architecture.

The local training lineage starts from `lerobot/smolvla_base`, uses the
`lerobot/libero` dataset, and proceeds through a usable 5k policy checkpoint, a
fresh 20k recovery job, and a fresh 50k continuation job.

## Current Results

The Stage 2 checkpoint was evaluated across all 10 LIBERO-Spatial tasks with
10 episodes per task. It completed 65 of 100 episodes successfully:

| Model | Scope | Successes | Success rate | 95% Wilson CI |
| --- | --- | ---: | ---: | ---: |
| Fine-tuned checkpoint, Stage 2 (50k run) | All 10 LIBERO-Spatial tasks | 65/100 | 65.0% | 55.3%--73.6% |
| Released SmolVLA LIBERO checkpoint | All 10 LIBERO-Spatial tasks | 80/100 | 80.0% | 71.1%--86.7% |

The following controlled task-0 comparison is retained to show the change
between training stages. It uses the same seed and 10 episodes per model.

| Model | Training stage | Successes | Success rate | 95% Wilson CI |
| --- | ---: | ---: | ---: | ---: |
| Released SmolVLA LIBERO checkpoint | Reference | 10/10 | 100% | 72.2%--100% |
| Fine-tuned checkpoint | Stage 1 (20k run) | 2/10 | 20% | 5.7%--51.0% |
| Fine-tuned checkpoint | Stage 2 (50k run) | 6/10 | 60% | 31.3%--83.2% |

Stage 2 improves the observed task-0 success rate by 40 percentage points over
Stage 1. The 100-episode suite result provides a more stable estimate than the
task-0 comparison. Under the same full-suite protocol, the fine-tuned model is
15 percentage points below the released checkpoint. This comparison shows a
clear gain from continued local training while retaining an honest reference
for the remaining performance gap.

The local policy reaches 81.25% of the released checkpoint's success rate
(65/80). This is a useful single-GPU result across the complete ten-task suite,
not a success limited to one selected task.

Per-task successes under the same 10-episode protocol are:

| Task ID | Fine-tuned Stage 2 | Released reference |
| ---: | ---: | ---: |
| 0 | 6/10 | 10/10 |
| 1 | 8/10 | 10/10 |
| 2 | 8/10 | 9/10 |
| 3 | 6/10 | 9/10 |
| 4 | 6/10 | 7/10 |
| 5 | 2/10 | 1/10 |
| 6 | 7/10 | 10/10 |
| 7 | 8/10 | 8/10 |
| 8 | 7/10 | 8/10 |
| 9 | 7/10 | 8/10 |

The fine-tuned policy matches or exceeds the reference on tasks 5 and 7. The
largest observed gap is four successes on task 0, followed by three-success
gaps on tasks 3 and 6. Because each per-task estimate uses only ten episodes,
these differences are descriptive rather than strong statistical claims.

## Demo Rollouts

The following videos are representative closed-loop rollouts from task 0 of
the Stage 2 full-suite evaluation. Both use the fine-tuned checkpoint and the
same evaluation protocol reported above.

- [Successful rollout](assets/stage2_task0_success.mp4) — the policy completes
  the LIBERO task and receives a success reward.
- [Failure rollout](assets/stage2_task0_failure.mp4) — a representative episode
  in which the policy does not satisfy the task predicate before termination.

The failure video is retained alongside the successful example to show the
observed behavior honestly rather than presenting only a selected success.

## Project Structure

```text
.
|-- assets/                         # Selected figures and short demo videos
|-- artifacts/                      # Small exported logs/configs (weights ignored)
|-- configs/
|   `-- evaluation.env.example     # Reproducible evaluation settings
|-- docs/
|   |-- DOWNLOAD_FROM_SERVER.md    # Files to retrieve after evaluation
|   |-- EXPERIMENTS.md             # Experiment protocol and interpretation
|   |-- GITHUB_UPLOAD.md           # Safe repository upload workflow
|   |-- HUGGINGFACE_MODEL_CARD.md  # Model card for the published checkpoint
|   |-- EVALUATION_GUIDE.md        # Chinese-first bilingual evaluation guide
|   |-- TRAINING_CONFIGURATION.md  # Verified training lineage and parameters
|   |-- TECHNICAL_REPORT_ZH.md     # Detailed Chinese technical report
|   `-- TECHNICAL_REPORT_EN.md     # Matching English technical report
|-- results/
|   |-- robustness_summary.csv     # Offline corruption experiment results
|   |-- spatial_per_task_summary.csv # Per-task fine-tuned/reference comparison
|   |-- task0_summary.csv          # Confirmed task-0 results
|   `-- spatial_suite_summary.csv  # Confirmed full-suite result and reference status
|-- scripts/
|   |-- evaluate_libero.sh         # Task-level or suite-level evaluation
|   |-- export_server_artifacts.sh # Build a small, GitHub-safe export bundle
|   `-- verify_checkpoint.sh       # Validate a LeRobot checkpoint
|-- .gitignore
|-- LICENSE
`-- README.md
```

## Environment

The tested environment uses:

- Ubuntu/Linux server
- NVIDIA RTX 4090 (24 GB)
- Python 3.12.14
- PyTorch 2.11.0+cu128 with CUDA 12.8
- LeRobot 0.6.2 at commit `9a6bb61043bac8c14353fcb6ea513b7473c118e3`
- CUDA-enabled PyTorch
- LIBERO and MuJoCo with EGL rendering

Install LeRobot and its LIBERO dependencies according to the upstream project,
then install the simulator extra from the LeRobot checkout:

```bash
conda activate smolvla
cd "$HOME/lerobot"
python -m pip install -e ".[libero]"
```

The exact package list and LeRobot commit used for the reported experiment
should be copied into `artifacts/environment/` before the repository is
published.

## Checkpoint Validation

```bash
MODEL_DIR=/absolute/path/to/checkpoint/pretrained_model \
  bash scripts/verify_checkpoint.sh
```

The validation script checks for the policy configuration, processor metadata,
and non-empty SafeTensors model files without loading the full model.

## Evaluation

Copy the example configuration and edit `MODEL_PATH`:

```bash
cp configs/evaluation.env.example configs/evaluation.env
```

Evaluate task 0 for 10 episodes:

```bash
bash scripts/evaluate_libero.sh configs/evaluation.env
```

Evaluate all tasks in the LIBERO-Spatial suite by setting the following in the
configuration file:

```bash
TASK_IDS=""
N_EPISODES=10
RUN_NAME="stage2_smolvla_spatial_all"
```

The policy used in this project expects three camera slots, while LIBERO emits
two image observations. Evaluation therefore preserves one empty camera slot
and maps the two simulator image keys to `camera1` and `camera2`.

## Reproducibility Notes

- All compared policies use seed `42`.
- The LIBERO control mode remains the default relative control mode.
- `policy.n_action_steps` is fixed to `10`.
- Task-0 comparisons use exactly the same task ID and initial-state sequence.
- Full-suite results should be reported separately from the preliminary
  single-task result.
- Model weights, optimizer states, datasets, caches, and complete video folders
  are intentionally excluded from Git. The final policy is hosted separately
  on Hugging Face.

## Limitations

- Each per-task estimate uses ten episodes and therefore has a wide confidence
  interval, although the overall comparison covers 100 episodes.
- The released reference model and the fine-tuned model may differ in training
  compute and data exposure, so the comparison is not a controlled ablation of
  model architecture.
- The robustness study is offline rather than a corrupted closed-loop rollout.
- Simulation success does not directly establish real-robot performance.

## License

The original code in this repository is released under the MIT License. Model,
dataset, simulator, and upstream LeRobot components remain subject to their own
licenses.
