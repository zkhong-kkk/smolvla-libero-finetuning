# Files to Download from the Training Server

Do not copy the entire output directory into this Git repository. Most of it is
either reproducible, too large for GitHub, or contains model weights that should
be hosted separately.

## Required Files

Download the following after the full evaluation finishes:

1. Training logs
   - Stage 1 `train.log`
   - Stage 2 `train.log`
2. Evaluation logs
   - Stage 1 task-0 evaluation log, if available
   - Stage 2 task-0 10-episode `eval.log`
   - Stage 2 full LIBERO-Spatial `eval.log`
   - Released-model full LIBERO-Spatial `eval.log`, after running the reference
     comparison
3. Small checkpoint metadata
   - `train_config.json`
   - policy `config.json`
   - `policy_preprocessor.json`
   - `policy_postprocessor.json`
   - normalizer metadata/configuration files
   - final `training_step.json`
4. Environment metadata
   - `pip freeze`
   - Conda environment export
   - NVIDIA driver/GPU information
   - LeRobot Git commit
5. Selected media
   - one successful Stage 2 task-0 video;
   - one failed Stage 2 task-0 video;
   - optionally one representative video from another Spatial task.

Place logs and configurations under `artifacts/`. Place only short,
representative videos under `assets/`.

## Do Not Put in Normal Git

- `model.safetensors` or sharded model weights;
- optimizer state;
- the complete checkpoint tree;
- Hugging Face cache;
- LIBERO assets or datasets;
- all 100 evaluation videos;
- Conda environment directories.

Use Hugging Face Hub, GitHub Releases, or Git LFS if model weights must be
published. Hugging Face Hub is the preferred option for a LeRobot checkpoint.
The final Stage 2 checkpoint for this project is already published at
[zkhomh-kkk/smolvla-libero-spatial-stage2](https://huggingface.co/zkhomh-kkk/smolvla-libero-spatial-stage2).

## Expected Local Layout

```text
artifacts/
|-- environment/
|   |-- conda_environment.yml
|   |-- gpu_info.txt
|   |-- lerobot_commit.txt
|   `-- pip_freeze.txt
|-- evaluation/
|   |-- official_task0_eval.log
|   |-- stage1_task0_eval.log
|   |-- stage2_task0_eval.log
|   `-- stage2_spatial_all_eval.log
|-- policy_metadata/
|   |-- config.json
|   |-- policy_postprocessor.json
|   |-- policy_preprocessor.json
|   |-- train_config.json
|   `-- training_step.json
`-- training/
    |-- stage1_train.log
    `-- stage2_train.log

assets/
|-- stage2_task0_success.mp4
`-- stage2_task0_failure.mp4
```

The included `scripts/export_server_artifacts.sh` creates this small export
bundle on the server without copying model weights.
