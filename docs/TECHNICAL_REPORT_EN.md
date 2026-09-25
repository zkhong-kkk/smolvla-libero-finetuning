# SmolVLA Fine-Tuning and LIBERO Evaluation on a Single RTX 4090

## Abstract

This project builds an end-to-end workflow for fine-tuning and evaluating the SmolVLA vision-language-action policy on a single NVIDIA RTX 4090. The work starts with a small training smoke test, continues with two fine-tuning stages on LIBERO data, and evaluates the resulting policies through closed-loop control in LIBERO-Spatial. The fine-tuned checkpoints and the released reference checkpoint are compared under the same task, random seed, action execution horizon, and camera mapping.

Two closed-loop comparisons are confirmed. On task 0, Stage 1 succeeds in 2/10 episodes (20%), Stage 2 succeeds in 6/10 (60%), and the released checkpoint succeeds in 10/10 (100%). In the full-suite evaluation, Stage 2 succeeds in 65/100 episodes across all ten LIBERO-Spatial tasks, for 65.0% with a 95% Wilson interval of 55.3%–73.6%. Under the identical protocol, the released checkpoint succeeds in 80/100, for 80.0% with an interval of 71.1%–86.7%. The local model is therefore 15 percentage points below the released reference while still showing a clear gain from continued training.

An exploratory offline robustness experiment was also performed using brightness changes, occlusion, noise, blur, and camera failure. Repeated action sampling was used as an uncertainty estimate. The policy was relatively stable under noise and blur but was more sensitive to severe brightness changes and camera failure. This experiment is an action-level proxy and must not be interpreted as closed-loop simulator success.

## 1. Motivation and Objectives

A vision-language-action policy must combine visual observations, language instructions, and robot state to produce continuous control actions. Training such a model from scratch is usually data- and compute-intensive. This project instead starts from released SmolVLA weights and studies what can be achieved with limited single-GPU resources.

The objectives are:

1. run a complete SmolVLA fine-tuning pipeline on one 24 GB RTX 4090;
2. connect the trained policy to LIBERO and MuJoCo for closed-loop evaluation;
3. compare Stage 1, Stage 2, and released reference checkpoints under controlled settings;
4. examine action stability and predictive uncertainty under visual corruption;
5. package configurations, scripts, logs, and results into a reproducible GitHub project.

### 1.1 Relationship to the Released Work

This project builds on the public SmolVLA, LeRobot, and LIBERO projects. It does not claim the released model architecture, dataset, or simulator as original work. Its contribution is a reproducible continuation study under limited hardware, with controlled comparisons across training stages and an additional visual-robustness analysis.

| Area | What the public projects provide | What this project adds |
| --- | --- | --- |
| Model | SmolVLA architecture, released weights, and basic inference capability | two-stage continued fine-tuning on one RTX 4090 and comparison of locally produced checkpoints |
| Framework | LeRobot training, processing, and the base `lerobot-eval` interface | experiment-specific entry points, checkpoint checks, log preservation, and artifact export |
| Simulator | LIBERO tasks, scenes, initial states, and success predicates | camera-interface adaptation and verified closed-loop rollouts for the selected policy |
| Baseline | released `lerobot/smolvla_libero` checkpoint | controlled comparison against local Stage 1 and Stage 2 checkpoints with identical rollout settings |
| Training analysis | a released final policy, not the behavior of this local continuation run | measured task-0 change from 20% to 60% and obtained a controlled full-suite comparison of 65% local versus 80% released |
| Robustness | the standard pipeline does not contain this exact offline corruption study | five corruption types, three severities, repeated action sampling, and a P90 uncertainty trigger |
| Deliverable | a general upstream codebase | bilingual reports, a unified evaluation guide, result CSV files, and a GitHub-safe project layout |

The main differences are therefore:

1. **training-stage comparison:** the project produces and evaluates local 20k and 50k checkpoints instead of only running the released policy;
2. **controlled closed-loop comparison:** task, initial-state sequence, action horizon, and camera mapping are fixed across local and released policies;
3. **visual robustness analysis:** the study extends beyond success rate to action drift, uncertainty, and fallback behavior under corruption;
4. **single-GPU reproducibility:** training, validation, evaluation, and export are organized into a workflow that can be reproduced on an RTX 4090.

These are experimental, evaluation, and engineering contributions. They are not a new VLA architecture or a state-of-the-art algorithm claim.

## 2. Technical Pipeline

```text
Released SmolVLA weights
        |
        v
Training smoke test -> two-stage fine-tuning -> checkpoint validation
                                                   |
                                                   v
                                LIBERO-Spatial closed-loop evaluation
                                                   |
                                                   v
                         reference comparison and confidence intervals

Exploratory branch: corrupt dataset images -> repeated action sampling
                    -> uncertainty and action-drift analysis
```

Fine-tuning optimizes the policy on demonstration data, while closed-loop evaluation tests whether sequential actions actually complete the simulated task. A lower training loss or single-step action error does not guarantee rollout success because small errors may accumulate over time.

## 3. Policy Interface

The released LIBERO configuration uses `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` as its vision-language backbone. The action dimension is 7. The policy predicts chunks of 50 actions and executes 10 action steps per inference call during evaluation through `policy.n_action_steps=10`.

LIBERO provides two RGB views:

- `agentview_image`, an external scene view;
- `robot0_eye_in_hand_image`, a wrist-camera view.

The policy expects three camera slots named `camera1`, `camera2`, and `camera3`. Evaluation therefore uses the following adapter:

```text
observation.images.image  -> observation.images.camera1
observation.images.image2 -> observation.images.camera2
camera3                   -> empty camera slot
```

This adapter changes only the feature interface, not the policy weights. Without it, LeRobot correctly rejects the environment-policy pair because the visual feature names do not match.

## 4. Experimental Environment

| Component | Configuration |
| --- | --- |
| Training GPU | one NVIDIA RTX 4090, 24 GB |
| Other server GPUs | four RTX 3090 cards, unused by this run |
| Operating system | Linux server |
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime | 12.8 |
| NVIDIA driver | 580.178.04 |
| Training and evaluation | LeRobot 0.6.2 |
| LeRobot commit | `9a6bb61043bac8c14353fcb6ea513b7473c118e3` |
| Simulation | LIBERO, MuJoCo, Robosuite |
| Headless rendering | EGL |
| Training device | one GPU, `CUDA_VISIBLE_DEVICES=0` |
| Evaluation seed | 42 |

The hardware, software versions, and code revision are stored in `artifacts/environment/ENVIRONMENT.txt`. A full `pip freeze` and Conda export can be added before publication.

## 5. Training

### 5.1 Smoke Test

A 100-step smoke test on an earlier small dataset first verified data loading, forward and backward passes, and checkpoint serialization. It used batch size 2 and reported approximately 450 million total parameters with 393 million trainable. It is not the final LIBERO configuration: both formal training stages use batch size 4, so the smoke-test loss and memory are not presented as final results.

The smoke test was used only for engineering validation. Its loss is not used as a final performance result.

### 5.2 Stage 1

The training lineage starts from `lerobot/smolvla_base` on the `lerobot/libero` dataset. An initial 25,000-step run reached step 5,000 but failed while saving a checkpoint because the device ran out of disk space:

```text
SafetensorError: I/O error: No space left on device
```

The 5k checkpoint did not contain a fully recoverable optimizer state, but its `pretrained_model` weights were usable. After disk cleanup, those weights initialized a fresh 20,000-step job with a new optimizer and scheduler. The job used batch size 4, learning rate `1e-4`, 500 warmup steps, and a 20k decay horizon. It finished in approximately 1 hour 31 minutes at 3.66 steps per second.

The Stage 1 checkpoint achieved 2/10 successes on task 0.

### 5.3 Stage 2

Stage 2 initialized from the valid Stage 1 checkpoint and ran a fresh 50,000-step job. It halved the peak learning rate from `1e-4` to `5e-5`, increased warmup from 500 to 1,000, and extended the decay horizon from 20k to 50k while keeping batch size 4.

After training, a one-episode smoke evaluation produced 0/1 success, while the formal ten-episode run produced 6/10. This contrast demonstrates that one rollout is useful for checking whether the pipeline runs, but it is not a reliable performance estimate.

### 5.4 What Was Actually Tuned

The project did not change the SmolVLA architecture or introduce a new Transformer block. Fine-tuning starts from `lerobot/smolvla_base`, updates the existing weights on `lerobot/libero`, and changes the training duration and learning-rate schedule.

#### A. Settings that update model weights

| Item | Project setting | Purpose |
| --- | --- | --- |
| Dataset | `lerobot/libero` in both stages | keeps the data source fixed |
| Weight lineage | 5k weights → Stage 1 20k → Stage 2 50k | about 75k inherited weight updates, but not a continuous optimizer resume |
| `freeze_vision_encoder` | `false` | allows the visual representation to adapt |
| `train_expert_only` | `false` | trains a broader parameter set than the action expert alone |
| `train_state_proj` | `true` | trains the robot-state projection |
| `use_peft` | `false` | no LoRA or other parameter-efficient adapter is used |
| Trainable parameters | about 393M of 450M in the smoke-test log | indicates broad joint fine-tuning rather than a small output head |
| Stage 1 | batch 4, LR `1e-4`, warmup 500, decay 20k | fresh 20k recovery job |
| Stage 2 | batch 4, LR `5e-5`, warmup 1,000, decay 50k | longer continuation with smaller updates |
| Shared optimizer settings | betas `[0.9,0.95]`, eps `1e-8`, weight decay `1e-10`, grad clip `10.0` | unchanged |
| AMP records | `policy.use_amp=true`, trainer `mixed_precision=no` | policy AMP path enabled without global trainer mixed precision |

The verified Stage 2 changes are therefore: 20k to 50k job length, `1e-4` to `5e-5` learning rate, 500 to 1,000 warmup steps, and 20k to 50k decay horizon. Data, batch size, trainable scope, and the remaining optimizer settings stay fixed. This is not a one-variable ablation, so the result supports the complete continuation recipe rather than assigning the gain to one parameter.

The full verified configuration and weight lineage are documented in [TRAINING_CONFIGURATION.md](TRAINING_CONFIGURATION.md).

#### B. Evaluation-only settings

| Item | Value | Meaning |
| --- | --- | --- |
| `chunk_size` | 50 | full action chunk predicted by the policy |
| `n_action_steps` | 10 | actions executed before observing and replanning |
| `seed` | 42 | fixes the initial-state sequence for comparison |
| `empty_cameras` | 1 | supplies the third expected camera slot |
| `rename_map` | `image/image2 -> camera1/camera2` | aligns environment and policy feature names |

These values do not retrain the model. In particular, `n_action_steps` is a control-frequency choice and the camera mapping is an interface adapter, not an architectural modification.

#### C. Robustness-analysis settings

The offline study uses 50 frames, three action samples per frame, and the 90th percentile of clean uncertainty as the trigger threshold. These choices affect measurement and fallback triggering only; they do not update model weights.

## 6. Closed-Loop LIBERO Evaluation

### 6.1 Controlled Setup

| Setting | Value |
| --- | --- |
| Suite | `libero_spatial` |
| Task ID | 0 |
| Episodes per policy | 10 |
| Seed | 42 |
| Batch size | 1 |
| Parallel environments | 1 |
| Asynchronous environments | disabled |
| Control mode | LIBERO default relative control |
| Executed action steps | 10 |
| Camera interface | two real views and one empty slot |

The metric is episode success as reported by the environment. It captures perception, control, and accumulated rollout error, and is therefore more informative than offline action MAE for deployment-like behavior.

The closed-loop procedure is:

1. LIBERO resets the selected task with its predefined initial state and seed;
2. the environment returns the external view, wrist view, robot state, and language task;
3. LeRobot maps camera keys and applies the checkpoint's image and state preprocessors;
4. SmolVLA predicts a 50-step action chunk, which is denormalized into the environment action space;
5. only the first ten actions are executed;
6. the updated environment is observed and the policy replans;
7. the loop continues until the LIBERO success predicate is satisfied or the episode reaches its limit.

This is receding-horizon control. Replanning after ten actions lets the policy use new visual feedback to correct previous errors. Ground-truth dataset actions are not supplied during evaluation, so there is no teacher forcing. Any error caused by an earlier action changes the next observation and can accumulate over the rollout.

Fair comparison does not require identical trajectories. It requires identical tasks, initial-state sequences, and control settings. The fixed task ID and `seed=42` provide that controlled starting point.

### 6.2 Confirmed Results

| Policy | Stage | Successes | Success rate | 95% Wilson interval |
| --- | --- | ---: | ---: | ---: |
| Released SmolVLA LIBERO | Reference | 10/10 | 100% | 72.2%–100% |
| Fine-tuned SmolVLA | Stage 1, 20k | 2/10 | 20% | 5.7%–51.0% |
| Fine-tuned SmolVLA | Stage 2, 50k | 6/10 | 60% | 31.3%–83.2% |

Stage 2 adds four successful episodes and improves the observed success rate by 40 percentage points over Stage 1. This is a useful engineering signal because the improvement appears in closed-loop task completion, not only in the training loss.

The result must still be interpreted conservatively:

- ten episodes per policy produce wide confidence intervals;
- the Stage 1 and Stage 2 intervals overlap;
- the confirmed comparison currently covers only task 0;
- the released reference still reaches 10/10;
- simulator success does not establish real-robot performance.

### 6.3 Full-Suite Result

Stage 2 has completed evaluation over all ten LIBERO-Spatial tasks, with ten episodes per task and 100 episodes in total.

| Policy | Scope | Successes | Success rate | 95% Wilson interval |
| --- | --- | ---: | ---: | ---: |
| Fine-tuned SmolVLA, Stage 2 50k | all 10 LIBERO-Spatial tasks | 65/100 | 65.0% | 55.3%–73.6% |
| Released SmolVLA LIBERO | all 10 LIBERO-Spatial tasks | 80/100 | 80.0% | 71.1%–86.7% |

This sample is ten times larger than the task-0 comparison and provides a more stable estimate. With task scope, episode count, seed, and control settings fixed, the released checkpoint records 15 additional successes, a 15-percentage-point gap. The Wilson intervals overlap slightly between 71.1% and 73.6%, so the report emphasizes the observed controlled gap rather than presenting it as a definitive architectural conclusion.

The result shows both that the local model works beyond task 0 and that the released checkpoint remains stronger. The local model reaches `65/80 = 81.25%` of the reference success rate. For a single-RTX-4090 undergraduate project, this is a solid result across all ten tasks rather than a success confined to one selected task. Possible causes of the remaining gap include data coverage, initialization, training duration, scheduling, and optimization stability; the current experiment cannot attribute it to one factor.

This report uses the full-suite 65.0% versus 80.0% comparison as its primary controlled result and retains the task-0 20%→60% change as local evidence of training-stage improvement.

### 6.4 Entry Point and Reproduction

The project uses one entry point: `bash scripts/evaluate_libero.sh configs/evaluation.env`. The local configuration selects the model, task scope, episode count, and output name. Set `TASK_IDS="[0]"` and `N_EPISODES=10` for the formal task-0 run; set `TASK_IDS=""` and `N_EPISODES=10` for the complete suite. For the released reference, change only `MODEL_PATH` to `lerobot/smolvla_libero` and keep the protocol fixed.

The script calls `lerobot-eval` and saves both `eval.log` and episode videos. Checkpoint validation, exact configuration examples, output layout, metric extraction, and the fair-comparison checklist are documented in [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).

## 7. Exploratory Offline Robustness Study

### 7.1 Design

This experiment predates the closed-loop LIBERO study and provides a low-cost view of action sensitivity. Fifty frames were sampled approximately uniformly from the dataset. Three action predictions were drawn per frame. Five corruptions were applied at light, medium, and heavy levels, producing 15 test conditions:

1. brightness change;
2. local occlusion;
3. image noise;
4. image blur;
5. camera failure.

### 7.2 Metrics

- **Clean MAE:** action error under the original image;
- **Corrupt MAE:** action error under the corrupted image;
- **Action drift:** change in the model output caused by corruption;
- **Uncertainty:** dispersion across repeated action samples;
- **Trigger rate:** fraction above the uncertainty threshold;
- **Fallback MAE:** error after applying the fallback rule;
- **Recovery:** relative error recovered by the fallback.

The uncertainty threshold was calibrated to the 90th percentile of uncertainty under clean inputs and was `0.6995` in this run.

The procedure has two stages. First, repeated predictions on clean images establish the policy's normal stochastic variation and define the P90 threshold. Second, the same frames are corrupted and evaluated again. If repeated predictions for a corrupted input are more dispersed than the threshold, the sample is marked uncertain and the fallback is triggered. Pairing clean and corrupted observations from the same frame reduces variation caused by different scene content.

### 7.3 Main Results

| Corruption | Severity | Corrupt MAE | Action drift | Trigger rate | Recovery |
| --- | --- | ---: | ---: | ---: | ---: |
| Brightness | Heavy | 0.9796 | 0.7123 | 22% | 13.90% |
| Occlusion | Heavy | 0.7965 | 0.5042 | 12% | 6.77% |
| Noise | Heavy | 0.5831 | 0.0520 | 12% | -0.49% |
| Blur | Heavy | 0.5932 | 0.1052 | 12% | 0.16% |
| Camera failure | Heavy | 1.0503 | 0.7783 | 22% | 15.86% |

The mean clean MAE was 0.5824. Severe camera failure and brightness shift caused the largest degradation. Severe noise and blur had relatively small effects in this offline sample.

Uncertainty and action drift had Pearson correlation 0.3983 and Spearman correlation 0.4640. These values indicate a moderate positive relationship, but uncertainty does not fully explain model degradation. The fallback recovered 15.86% of the error under severe camera failure and 13.90% under severe brightness shift, while negative recovery in some mild conditions shows that a simple trigger can intervene unnecessarily.

### 7.4 Scope

This is not an official LIBERO robustness benchmark. It modifies recorded images and measures action-level behavior without executing a full corrupted rollout. It supports claims about sensitivity to specific visual changes, but not claims about improved task success. A stronger follow-up would corrupt live LIBERO observations and compare closed-loop success with and without the fallback.

## 8. Key Engineering Issue: Camera Feature Adaptation

The main integration issue was the visual interface mismatch between the environment and policy. LIBERO emits two fields, `image` and `image2`, while the SmolVLA policy configuration expects `camera1`, `camera2`, and `camera3`. The project maps the two real views to the first two policy fields and sets `empty_cameras=1` for the third slot. This adapter satisfies LeRobot's feature-consistency validation and connects the policy to LIBERO without modifying model weights.

## 9. Contributions

The completed work includes:

- deploying and validating SmolVLA fine-tuning on a single RTX 4090;
- completing two training stages and handling disk-full, corrupted-checkpoint, and output-directory failures;
- integrating SmolVLA with LIBERO/MuJoCo for closed-loop evaluation;
- resolving the two-camera environment versus three-camera policy interface;
- building a controlled comparison of released, Stage 1, and Stage 2 policies;
- designing a 5-corruption, 3-severity offline robustness study;
- packaging reproducible evaluation, checkpoint validation, and artifact export scripts.

A concise resume bullet is:

> Fine-tuned SmolVLA in two stages on a single RTX 4090 using LeRobot and built a closed-loop LIBERO evaluation pipeline; the Stage 2 policy achieved 65% success over 100 episodes covering all ten LIBERO-Spatial tasks versus 80% for the released checkpoint, while the controlled task-0 comparison improved from 20% to 60%, with additional action-stability analysis under visual corruption.

The 65% versus 80% result is the full-suite comparison, while 20%→60% is the ten-episode task-0 stage comparison. Their scopes should remain explicit.

## 10. Limitations and Next Steps

Current limitations are the ten-episode sample size for each individual task,
offline rather than closed-loop robustness testing, the absence of real-robot
data, and the lack of controlled hyperparameter ablations.

The archived per-task success counts for task IDs 0--9 are
`6, 8, 8, 6, 6, 2, 7, 8, 7, 7` for the fine-tuned policy and
`10, 10, 9, 9, 7, 1, 10, 8, 8, 8` for the released reference. The fine-tuned
policy exceeds the reference by one success on task 5 and matches it on task 7.
The largest observed gap is four successes on task 0, followed by
three-success gaps on tasks 3 and 6. These per-task differences are diagnostic
rather than strong statistical claims.

The next priorities are:

1. inspect success and failure videos for tasks 0, 3, and 6;
2. inject brightness and camera-failure corruptions into live rollouts;
3. increase the episode count for the largest per-task gaps;
4. keep large weights on Hugging Face Hub and store only code, configs, small logs, and selected videos in Git.

The final Stage 2 checkpoint is published at
[zkhomh-kkk/smolvla-libero-spatial-stage2](https://huggingface.co/zkhomh-kkk/smolvla-libero-spatial-stage2).
It includes the SafeTensors weights, policy configuration, preprocessing and
postprocessing state, tokenizer, and training configuration required for
LeRobot loading.

## 11. Conclusion

The main value of this project is the complete engineering loop from fine-tuning and checkpoint management to simulator integration and controlled evaluation. Stage 2 succeeds in 65 of 100 episodes across all ten LIBERO-Spatial tasks, compared with 80 of 100 for the released checkpoint under the same protocol. The fixed task-0 comparison improves from 20% to 60%, demonstrating a practical gain from continued training while the 15-point full-suite gap shows the remaining room for improvement. The robustness study identifies camera failure and severe brightness shift as useful targets for future closed-loop experiments.
