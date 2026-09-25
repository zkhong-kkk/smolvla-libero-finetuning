# Evaluation Guide / 评测指南

## 中文

### 1. 评测入口

统一评测入口是：

```bash
bash scripts/evaluate_libero.sh configs/evaluation.env
```

脚本最终调用 LeRobot 官方命令 `lerobot-eval`，负责创建 LIBERO 环境、加载模型、运行闭环 episode、保存视频和汇总成功率。不要直接修改脚本中的参数；每次评测只修改本地配置文件 `configs/evaluation.env`。该文件已被 `.gitignore` 忽略，因此服务器绝对路径不会提交到 GitHub。

### 2. 第一次使用

在服务器的本项目目录中执行：

```bash
conda activate smolvla
cp configs/evaluation.env.example configs/evaluation.env
```

打开 `configs/evaluation.env`，把 `MODEL_PATH` 改成待评测模型的 `pretrained_model` 目录。例如：

```bash
MODEL_PATH="/mnt/disk2/hongzekun/outputs/your_stage2_run/checkpoints/last/pretrained_model"
```

官方模型可以直接写 Hugging Face 仓库 ID：

```bash
MODEL_PATH="lerobot/smolvla_libero"
```

### 3. 先检查 checkpoint

评测自训练模型前运行：

```bash
export MODEL_DIR="/mnt/disk2/hongzekun/outputs/your_stage2_run/checkpoints/last/pretrained_model"
bash scripts/verify_checkpoint.sh
```

输出 `CHECKPOINT_OK` 后再开始评测。这个检查能发现磁盘写满造成的空文件或不完整权重。

### 4. 单任务冒烟评测

冒烟评测只用于确认模型、环境和视频保存都能正常运行。设置：

```bash
TASK_SUITE="libero_spatial"
TASK_IDS="[0]"
N_EPISODES=1
RUN_NAME="stage2_task0_smoke"
```

然后运行统一入口：

```bash
bash scripts/evaluate_libero.sh configs/evaluation.env
```

1 个 episode 的结果不能用于比较模型性能。

### 5. 单任务正式评测

复现本项目任务 0 的结果时设置：

```bash
TASK_SUITE="libero_spatial"
TASK_IDS="[0]"
N_EPISODES=10
SEED=42
RUN_NAME="stage2_task0_10ep"
```

运行：

```bash
bash scripts/evaluate_libero.sh configs/evaluation.env
```

比较 Stage 1、Stage 2 和官方模型时，除 `MODEL_PATH` 和 `RUN_NAME` 外，其余配置必须保持不变。

### 6. LIBERO-Spatial 全任务评测

全任务评测覆盖 10 个任务，每个任务运行 10 个 episode，共 100 个 episode。设置：

```bash
TASK_SUITE="libero_spatial"
TASK_IDS=""
N_EPISODES=10
SEED=42
RUN_NAME="stage2_spatial_all_10ep"
```

运行相同入口：

```bash
bash scripts/evaluate_libero.sh configs/evaluation.env
```

评测自训练模型后，只修改以下两项再评测官方模型：

```bash
MODEL_PATH="lerobot/smolvla_libero"
RUN_NAME="official_spatial_all_10ep"
```

### 7. 固定评测方法

本项目所有可比较的闭环评测使用以下设置：

| 项目 | 固定值 |
| --- | --- |
| 环境 | LIBERO |
| 任务套件 | `libero_spatial` |
| 随机种子 | 42 |
| Batch size | 1 |
| 并行任务数 | 1 |
| 异步环境 | 关闭 |
| 动作执行步数 | 10 |
| 相机槽位 | 两个真实图像加一个空槽位 |
| 相机字段映射 | `image -> camera1`，`image2 -> camera2` |

环境在每个 episode 中给出观测，策略预测动作块，执行前 10 个动作后重新观察并预测，直到任务成功或达到 episode 上限。这是闭环评测：模型后续看到的状态由前面动作实际造成，而不是固定数据集中的下一帧。

更具体地说，模型一次生成 50 步动作，但控制器只执行前 10 步，然后丢弃剩余未执行部分并根据新观测重新生成动作。这相当于滚动时域控制。评测时不读取数据集真实动作，LIBERO 的任务完成条件是唯一的成功判定依据。

### 8. 指标与统计方法

主要指标为成功率：

```text
success_rate = n_success / n_episodes * 100%
```

任务成功由 LIBERO 环境的任务判定函数给出。报告同时保存成功次数、episode 总数和 95% Wilson 置信区间。Wilson 区间适合成功/失败这种二项结果，尤其适合本项目 10 个 episode 的小样本情况。

单任务结果只代表该任务。全套件总体成功率应由全部 100 个 episode 汇总，不能用任务 0 的成功率代替。

### 9. 输出位置

脚本在 `OUTPUT_ROOT` 下生成带时间戳的目录：

```text
OUTPUT_ROOT/
`-- RUN_NAME_YYYYMMDD_HHMMSS/
    |-- eval.log
    `-- videos/
        `-- libero_spatial_<task_id>/
            `-- eval_episode_<episode_id>.mp4
```

终端外还会先生成一个同名 `.log` 文件，评测成功后复制到结果目录中的 `eval.log`。

查看总体成功率：

```bash
grep -E "Output dir:|Success rate [0-9]" /absolute/path/to/eval.log
```

查看包含每任务结果的汇总行：

```bash
grep -E "Aggregated Metrics for per_task|pc_success" /absolute/path/to/eval.log
```

确认结果后，将总体数据填写到 `results/spatial_suite_summary.csv`，并把代表性的成功与失败视频各选择一个放入 `assets/`。完整视频目录不要提交到 GitHub。

### 10. 公平对照检查表

在比较两个模型前确认：

- `TASK_SUITE`、`TASK_IDS` 和 `N_EPISODES` 相同；
- `SEED` 相同；
- `N_ACTION_STEPS` 相同；
- `RENAME_MAP` 和 `EMPTY_CAMERAS` 相同；
- 使用同一个 LeRobot commit 和 Python 环境；
- 不根据单次失败临时更换 seed；
- 同时报告成功次数、总次数和置信区间。

## English

### 1. Entry Point

All evaluations use one entry point:

```bash
bash scripts/evaluate_libero.sh configs/evaluation.env
```

The wrapper calls the official `lerobot-eval` command, creates the LIBERO environment, loads the policy, executes closed-loop episodes, saves videos, and reports success metrics. Edit `configs/evaluation.env`, not the shell script. The local configuration is ignored by Git so server-specific absolute paths are not published.

### 2. Initial Setup

```bash
conda activate smolvla
cp configs/evaluation.env.example configs/evaluation.env
```

Set `MODEL_PATH` to a local `pretrained_model` directory or to the released model ID:

```bash
MODEL_PATH="/absolute/path/to/checkpoints/last/pretrained_model"
# or
MODEL_PATH="lerobot/smolvla_libero"
```

Validate a local checkpoint first:

```bash
export MODEL_DIR="/absolute/path/to/checkpoints/last/pretrained_model"
bash scripts/verify_checkpoint.sh
```

### 3. Evaluation Modes

Task-0 smoke test:

```bash
TASK_IDS="[0]"
N_EPISODES=1
RUN_NAME="stage2_task0_smoke"
```

Task-0 formal evaluation:

```bash
TASK_IDS="[0]"
N_EPISODES=10
SEED=42
RUN_NAME="stage2_task0_10ep"
```

Full LIBERO-Spatial evaluation, ten episodes for each of ten tasks:

```bash
TASK_IDS=""
N_EPISODES=10
SEED=42
RUN_NAME="stage2_spatial_all_10ep"
```

After editing the configuration, run the same entry command. For a reference comparison, change only `MODEL_PATH` to `lerobot/smolvla_libero` and choose a new `RUN_NAME`.

### 4. Protocol and Metric

Comparable runs keep seed 42, batch size 1, one synchronous environment, ten executed action steps, one empty camera slot, and the same image-key mapping. The policy repeatedly observes the simulator, predicts an action chunk, executes ten actions, and observes again. Success is determined by the LIBERO task predicate.

Report `n_success`, `n_episodes`, success rate, and the 95% Wilson interval. A one-episode smoke test is only a runtime check. Task-0 success must not be reported as the full-suite score.

### 5. Outputs

Each run creates a timestamped directory containing `eval.log` and task-specific videos. Read the main result with:

```bash
grep -E "Output dir:|Success rate [0-9]" /absolute/path/to/eval.log
```

Copy confirmed aggregate values to `results/spatial_suite_summary.csv`. Keep only selected success and failure videos in Git.
