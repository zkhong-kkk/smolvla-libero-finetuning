---
license: apache-2.0
library_name: lerobot
tags:
  - robotics
  - lerobot
  - smolvla
  - vla
  - libero
base_model: lerobot/smolvla_base
datasets:
  - lerobot/libero
---

# SmolVLA LIBERO-Spatial Stage 2

This repository contains the final SmolVLA policy produced by a single-GPU
two-stage fine-tuning project using LeRobot and the LIBERO dataset.

## Model

- Hub ID: `zkhomh-kkk/smolvla-libero-spatial-stage2`
- Starting checkpoint: `lerobot/smolvla_base`
- Dataset: `lerobot/libero`
- Evaluation suite: `libero_spatial`
- Hardware: one NVIDIA RTX 4090 with 24 GB VRAM
- LeRobot version: 0.6.2
- LeRobot commit: `9a6bb61043bac8c14353fcb6ea513b7473c118e3`

The final weights inherit approximately 75k updates through three jobs: a
usable 5k policy checkpoint, a fresh 20k Stage 1 job initialized from those
weights, and a fresh 50k Stage 2 job initialized from Stage 1. This is not a
continuous 75k optimizer resume because each new job recreated the optimizer
and scheduler.

Stage 2 used batch size 4, peak learning rate `5e-5`, 1,000 warmup steps, and a
50,000-step decay horizon. The vision encoder was trainable, expert-only
training was disabled, and PEFT was not used.

## Evaluation Results

Both this checkpoint and the released `lerobot/smolvla_libero` reference were
evaluated with the same LIBERO-Spatial protocol: all ten tasks, ten episodes per
task, seed 42, batch size 1, synchronous environments, and ten executed action
steps per policy call.

| Policy | Successes | Success rate | 95% Wilson CI |
| --- | ---: | ---: | ---: |
| This Stage 2 checkpoint | 65/100 | 65.0% | 55.3%--73.6% |
| Released reference | 80/100 | 80.0% | 71.1%--86.7% |

Per-task successes for task IDs 0--9 are
`6, 8, 8, 6, 6, 2, 7, 8, 7, 7`. The complete protocol, logs, and comparison
tables are available in the associated GitHub project:
[zkhong-kkk/smolvla-libero-finetuning](https://github.com/zkhong-kkk/smolvla-libero-finetuning).

## Using the Checkpoint

LeRobot can resolve the policy directly from its Hub ID:

```bash
MODEL_PATH="zkhomh-kkk/smolvla-libero-spatial-stage2"
```

For LIBERO evaluation, the environment image keys must be adapted to the policy
camera keys:

```bash
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
EMPTY_CAMERAS=1
```

The GitHub repository provides the complete `lerobot-eval` wrapper and
configuration example.

## Limitations

- The checkpoint was evaluated in simulation and has not been validated on a
  physical robot.
- Each individual task has only ten evaluation episodes.
- The visual-corruption experiment in the project is an offline action-level
  analysis, not a closed-loop corrupted LIBERO benchmark.
- The checkpoint does not outperform the released SmolVLA LIBERO reference.

## License and Attribution

The checkpoint is derived from the released SmolVLA model and LIBERO training
data. Users should follow the licenses and terms of SmolVLA, LeRobot, LIBERO,
and the underlying dataset. The model repository declares Apache-2.0 to match
the released upstream checkpoint.
