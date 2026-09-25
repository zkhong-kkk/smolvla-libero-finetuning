# Training Configuration / 训练配置

## 中文

### 1. 训练链路

最终模型不是一次从头连续训练 50,000 step 得到的，而是经过三段权重继承：

```text
`lerobot/smolvla_base` 初始化
        |
        v
最初训练到 5,000 step
（保存时磁盘写满，优化器状态不完整，但 pretrained_model 权重可用）
        |
        v
Stage 1：以 5k 权重初始化新的 20,000-step 训练任务
（重新创建优化器和学习率调度器）
        |
        v
Stage 2：以 Stage 1 的 20k checkpoint 初始化新的 50,000-step 训练任务
（再次创建优化器和学习率调度器）
        |
        v
最终 Stage 2 模型
```

因此，最终权重经历了大约 `5k + 20k + 50k = 75k` 次训练更新。但这不是一次连续的 75k resume：每个新任务会重新初始化优化器和学习率调度器。报告中用“Stage 1 20k job”和“Stage 2 50k job”表示各训练任务自身的步数，用“约 75k 权重训练链路”表示最终模型继承的累计更新历史。

### 2. 两阶段真实配置

以下数值来自两个最终 checkpoint 保存的 `train_config.json` 和 `training_step.json`。

| 参数 | Stage 1：20k recovery job | Stage 2：50k job | 变化与目的 |
| --- | ---: | ---: | --- |
| 初始模型 | `lerobot/smolvla_base` 经最初 5k 更新后的权重 | Stage 1 的 20k checkpoint | 明确权重继承来源 |
| 数据集 | `lerobot/libero` | `lerobot/libero` | 保持数据来源一致 |
| 初始化权重 | 最初运行的 5k `pretrained_model` | Stage 1 的 20k checkpoint | 逐阶段继承已学习权重 |
| `steps` | 20,000 | 50,000 | 延长第二阶段优化时间 |
| `batch_size` | 4 | 4 | 保持单步 batch 一致 |
| `grad_accum_steps` | 1 | 1 | 无额外梯度累积 |
| 有效 batch size | 4 | 4 | 单 GPU：4 × 1 |
| `num_workers` | 4 | 4 | 保持数据加载设置一致 |
| `seed` | 42 | 42 | 保持随机性设置一致 |
| `save_freq` | 10,000 | 25,000 | Stage 2 减少大 checkpoint 保存次数 |
| `optimizer_lr` | `1e-4` | `5e-5` | 第二阶段将峰值学习率减半，进行更稳健的继续微调 |
| `optimizer_betas` | `[0.9, 0.95]` | `[0.9, 0.95]` | 不变 |
| `optimizer_eps` | `1e-8` | `1e-8` | 不变 |
| `optimizer_weight_decay` | `1e-10` | `1e-10` | 不变 |
| `optimizer_grad_clip_norm` | `10.0` | `10.0` | 不变，限制异常梯度 |
| `scheduler_warmup_steps` | 500 | 1,000 | Stage 2 使用更长的学习率预热 |
| `scheduler_decay_steps` | 20,000 | 50,000 | 与各阶段训练长度对应 |
| `scheduler_decay_lr` | `2.5e-6` | `2.5e-6` | 最终衰减目标不变 |

Stage 2 相较 Stage 1 的核心调参只有四项：

1. 新任务训练步数从 20k 增加到 50k；
2. 初始学习率从 `1e-4` 降到 `5e-5`；
3. warmup 从 500 增加到 1,000；
4. decay horizon 从 20k 延长到 50k。

这套调整的逻辑是：第二阶段从已有 checkpoint 出发，不再需要和第一阶段一样大的更新幅度，因此将学习率减半；同时用更长的 warmup 和衰减周期配合更长的训练，降低继续训练对已有能力的破坏。

### 3. 训练了模型的哪些部分

| 参数 | 两阶段设置 | 含义 |
| --- | --- | --- |
| `freeze_vision_encoder` | `false` | 视觉编码器参与权重更新 |
| `train_expert_only` | `false` | 不只训练动作专家部分 |
| `train_state_proj` | `true` | 机器人状态投影层参与训练 |
| `use_peft` | `false` | 未使用 LoRA 或 adapter |
| `policy.use_amp` | `true` | 策略配置启用 AMP 路径 |
| 训练状态 `mixed_precision` | `no` | 分布式训练层没有设置全局 mixed-precision 模式 |

这不是只训练最后一层或 LoRA 的轻量微调，而是较大范围的联合微调。模型结构本身没有改变：视觉语言主干仍为 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，注意力模式仍为 `cross_attn`，VLM 层数为 16。

### 4. 保持不变的模型和动作配置

| 参数 | 设置 |
| --- | ---: |
| `chunk_size` | 50 |
| `n_action_steps` | 10 |
| 动作生成迭代数 `num_steps` | 10 |
| `max_state_dim` | 32 |
| `max_action_dim` | 32 |
| `empty_cameras` | 1 |
| `resize_imgs_with_padding` | `[512, 512]` |
| `attention_mode` | `cross_attn` |
| `num_vlm_layers` | 16 |
| `num_expert_layers` | 0 |

其中 `n_action_steps=10` 和 `empty_cameras=1` 同时用于评测，但它们不是 Stage 2 新调出来的参数。前者决定闭环执行频率，后者解决两路 LIBERO 图像与三相机策略接口的适配。

### 5. Checkpoint 信息

| 项目 | Stage 1 | Stage 2 |
| --- | --- | --- |
| 最终目录 | `checkpoints/020000` | `checkpoints/050000` |
| 模型文件 | `model.safetensors` | `model.safetensors` |
| 模型文件大小 | 864.71 MiB | 864.71 MiB |
| checkpoint 模型部分总大小 | 约 0.844 GiB | 约 0.844 GiB |
| 配置、预处理和后处理文件 | 完整 | 完整 |

### 6. 参数变化与最终效果

| 模型 | 范围 | 成功率 |
| --- | --- | ---: |
| Stage 1：5k 权重 + 20k recovery job | 任务 0，10 episodes | 20%（2/10） |
| Stage 2：再训练 50k | 任务 0，10 episodes | 60%（6/10） |
| Stage 2 最终模型 | 全 10 个任务，100 episodes | 65%（65/100） |
| 官方 SmolVLA LIBERO | 全 10 个任务，100 episodes | 80%（80/100） |

Stage 2 在任务 0 上比 Stage 1 提高 40 个百分点，说明“降低学习率并延长训练”获得了实际闭环收益。最终模型在完整套件上成功 65 次，达到官方模型成功率的 `65 / 80 = 81.25%`。

这个效果对于单张 RTX 4090 完成的本科项目是扎实的：它不是只在一个挑选任务上成功，而是在 10 个任务、100 个闭环 episode 上取得 65% 成功率。模型没有超过官方 checkpoint，因此报告保留 15 个百分点差距；但训练阶段提升和完整套件结果共同说明微调确实学到了可执行的策略，而不是只降低离线损失。

## English

### 1. Training Lineage

The final policy starts from `lerobot/smolvla_base` and inherits approximately 75k weight updates through three jobs: an initial 5k run, a fresh 20k Stage 1 job initialized from the usable 5k policy weights, and a fresh 50k Stage 2 job initialized from Stage 1. This is not a continuous 75k optimizer resume because the optimizer and scheduler were recreated for each new job.

### 2. Verified Stage Configuration

| Parameter | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| Initialization | 5k weights derived from `lerobot/smolvla_base` | Stage 1 20k checkpoint |
| Dataset | `lerobot/libero` | `lerobot/libero` |
| Job steps | 20,000 | 50,000 |
| Batch size | 4 | 4 |
| Gradient accumulation | 1 | 1 |
| Data workers | 4 | 4 |
| Seed | 42 | 42 |
| Save frequency | 10,000 | 25,000 |
| Peak learning rate | `1e-4` | `5e-5` |
| Warmup | 500 | 1,000 |
| Decay horizon | 20,000 | 50,000 |
| Final decay LR | `2.5e-6` | `2.5e-6` |

Stage 2 halves the learning rate, doubles warmup, and extends both job length and scheduler horizon. Both stages train the vision encoder and parameters beyond the action expert, train the state projection, and use no PEFT adapter.

### 3. Outcome

Task-0 success improves from 20% at Stage 1 to 60% at Stage 2. The final policy achieves 65/100 over all ten LIBERO-Spatial tasks, compared with 80/100 for the released checkpoint. It therefore reaches 81.25% of the reference success rate while retaining an honest 15-point performance gap.
