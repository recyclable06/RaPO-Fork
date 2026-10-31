# 全参数覆盖的 LoRA 训练设计

日期：2026-10-30。状态：首轮代码已实现，微型 Qwen2-VL 的 CPU 检查已通过；CUDA 多卡、vLLM 同步及真实实验尚未验证。

本文是本项工作的入口。[决策记录](decisions.md)记录重要取舍；[验证与实验计划](validation-plan.md)说明最小正确性检查和结果比较。

## 1. 要得到的结果

在现有 Qwen2-VL-2B-Instruct、ImageNet-R 类增量图像分类流程上，将全参数直接更新改为“矩阵及卷积核使用 LoRA，归一化和偏置向量直接训练”，使原来全部可训练参数的有效值都有更新途径。准确称为 **LoRA + norm/bias**，不称为每种参数都经过低秩分解的纯 LoRA。

保留同一套适配参数依次学习各任务、测试时判断全部已见类别的方式。保留 RaPO 的原始 Ref、上一任务 Anchor、Retention Reward 和 CTAN。新实验与已有全参数结果比较准确率、遗忘和资源消耗。

两组实验从同一原始模型版本开始。不得把已学完 ImageNet-R 的全参数 checkpoint 当作 LoRA 起点进行更新方式对照。

## 2. 已确认的范围与边界

- 用户要求覆盖全部参数，而非仅语言侧或仅 `all-linear`；向量采用直接训练。
- 模型的视觉、连接、语言部分均纳入覆盖检查；不因软件接口不支持而静默冻结某类参数。
- 实验室没有可直接复用的已验证实现；允许评估独立新环境。具体依赖组合仍待确定。
- 沿用现有仓库、训练入口、任务序列、日志和实验结果结构。交付可运行的实验、可用的模型结果及与全参数结果的比较。
- 本项不建设调度器、自动重试、监控服务、实验数据库、通用 adapter 路由或新分布式框架。
- 第一轮只增加正确训练、正确生成、正确任务切换和保存可用结果所必需的实现。恢复能力、内存优化和额外统计仅在实际需求出现时增加。

## 3. 参数处理契约

“覆盖”以注入适配器之前的原模型参数为基准，不以注入后可训练参数数量为基准；共享参数不能误算成两份独立权重。

| 原参数类别 | 更新方式 | 导出的有效权重 |
|---|---|---|
| 视觉、Merger、语言模块的线性矩阵 | 冻结原矩阵，训练 LoRA A/B | `W0 + (alpha/r) * B @ A` |
| 视觉 patch embedding 的卷积核 | 卷积 LoRA；按层的卷积定义合并 | 原卷积核 + 对应低秩增量 |
| 文本 embedding 矩阵 | Embedding LoRA | 原嵌入矩阵 + 对应低秩增量 |
| LM head 权重 | 保持原模型的共享关系；不得独立拆开共享更新 | 与共享 embedding 一致的有效权重 |
| 归一化权重及所有原有偏置向量 | 原参数直接训练 | 当前参数值 |
| 实际枚举发现的其他原有可训练参数 | 必须显式分类后处理 | 不允许自动忽略 |

非参数 buffer、tokenizer、提示词、KV cache 不属于此处的可训练参数覆盖。

Qwen2-VL-2B 官方配置启用了 `tie_word_embeddings`；以实际模型为准，适配和导出后保持同一有效权重服务两处。线性、卷积和 embedding 使用 PEFT；共享输出头用一个小模块直接复用 embedding 的 A/B，避免转置后另注册一份 Parameter。两处在同一个根 FSDP 单元中；仅 Transformer blocks 单独包装，使用 `use_orig_params=True` 支持冻结/可训练参数混合。

初始化时核对一次覆盖范围，输出各类更新方式的参数数量及未覆盖项即可。没有未覆盖项且共享关系正确后，不再逐轮打印参数；不开发逐参数审计报告或参数登记系统。

## 4. 模型角色与数据流

| 角色 | 状态 | 更新时机 |
|---|---|---|
| Actor | 冻结的原矩阵/卷积核 + 可训练 LoRA + 可训练 norm/bias | 每次优化器更新 |
| Ref | 独立冻结的原始模型 | 始终不变 |
| Anchor | 独立冻结的上一任务有效模型 | 新任务开始前更新一次 |
| vLLM rollout | 普通模型结构，装载 Actor 的有效权重 | 需要使用新策略生成或评估之前同步 |

同一份有效权重导出逻辑服务 vLLM 和 Anchor。它消解 PEFT 包装、合并低秩增量、包含当前向量值，并保留共享权重语义。

```text
Actor 更新
    └─ 导出有效权重 ──> vLLM 生成/评估

任务 t 结束
    └─ 导出任务 t 的有效权重 ──> Anchor 冻结供任务 t+1 使用
                                   Actor 原地继续训练，不重建 adapter/优化器

Ref 始终保持实验开始时的原始模型
```

### Ref 必须与当前 Actor 的向量更新隔离

当前 LoRA 路径通过 `disable_adapter()` 将 Actor 当作 Ref。直接训练 norm/bias 后，关闭 adapter 并不能恢复这些向量的初始值，因而不再代表原始策略。本方案采用独立 Ref，复用全参数路径的角色划分，避免增加参数来回恢复的机制。

### 有效权重导出

- 使用参数名、形状和明确的共享关系对应权重，不使用两个模型 `parameters()` 的位置顺序配对。
- 导出不得改变训练端被冻结的矩阵，不永久合并或删除 Actor 的 adapter，不重置其优化器。
- 使用固定输入检查合并后的输出数值，考虑 BF16 舍入；这用于发现漏层或重复增量，不追求不同后端位级一致。
- 使用 FSDP 支持的参数聚合/状态接口；各 rank 的集体通信顺序必须一致。具体聚合粒度在小规模验证后确定。
- 第一版使用标准接口实现最短的正确导出路径，允许必要的临时状态，用完释放。只有实测合并峰值超出资源时才做分块；不把流式导出列为前置工程。
- 每次同步更新完整有效策略。同步后不得复用旧权重计算出的 KV/视觉特征缓存；优先沿用现有按批次生成和缓存生命周期。
- vLLM 使用普通权重加载路径，不同时启用独立 LoRA 分支，防止同一增量被应用两次。

## 5. 环境路线与实现位置

新版 vLLM 已增加 Qwen2-VL 视觉/连接 LoRA 支持，但其独立 adapter 路径仍有 bias、额外可训练模块等限制。升级不能单独满足本文的全参数覆盖目标。

当前推荐：先检查现有 PEFT 是否满足所需层类型和共享权重；缺少能力时才在独立环境中调整必要依赖。优先保留已运行的训练栈与普通 vLLM 权重同步路径，不开展多版本兼容矩阵或整体框架迁移。若现成实现能直接省掉适配工作，优先复用它。

首轮实现的代码落点：

| 位置 | 职责 |
|---|---|
| `verl/workers/actor/config.py`、`verl/workers/fsdp_workers.py` | 配置、参数覆盖、优化器参数范围、独立 Ref、FSDP 包装 |
| `verl/utils/full_lora.py` | 参数覆盖、共享输出头、非原地有效权重合并；FSDP full state dict 聚合 |
| `verl/workers/sharding_manager/fsdp_vllm.py`、rollout 初始化 | 使用普通模型权重同步，消费导出结果 |
| `examples/baselines/img_cls_cil/image_cls_cil_rapo.py` | 用有效权重更新 Anchor，继承适配器与优化器状态 |
| `verl/utils/checkpoint/fsdp_checkpoint_manager.py` | 优先沿用保存路径；仅在漏存新参数或不能读取结果时修改 |
| 现有启动脚本/配置 | 选择全参数或新方案、一个 seed 和实际训练配置 |

FSDP 需要处理冻结矩阵与可训练向量混合。优先选用标准支持的包装方式/配置；先验证正确性与实际内存，再判断是否值得细化包装。不得以静默冻结向量规避问题。

## 6. 保存可用结果

沿用现有 checkpoint 和 CTAN/任务结果保存，不改格式、不另写保存管理器。已有优化器等状态继续按原流程保存；重点只检查新增的 LoRA、norm/bias 没有漏掉。完成一次模型重载，确认产物可用于评估即可。

若需要导出合并后的普通模型，复用有效权重逻辑；它是推理产物，不宣称能还原 LoRA 分解和优化器状态。

任务边界续训、任务中途恢复、自动回滚及故障注入均不列为首轮实现或验收要求。若之后确实需要续训，再针对具体恢复位置处理 Anchor、CTAN、优化器和数据位置；未验证前不承诺恢复能力。正常连续运行中的任务切换仍是 RaPO 核心业务，必须正确。

## 7. 实验配置与待测项

- 同一原始模型版本、数据划分、类别顺序、提示词、奖励设置、全局 batch、rollout 数量和训练轮数作为对照基础。
- rank、alpha、学习率仍是待选实验参数；可先用单一 rank 做工程验证，不能把候选值标成已验证的最优配置。
- 第一轮先保持已跑通的卡数和 micro-batch，避免同时更换更新方式与并行配置。工程正确后再评价是否能减少卡数。
- 参数与优化器开销预计减少，但完整权重同步、Ref/Anchor、激活和生成缓存仍存在。优先从已有日志取得耗时和显存，并用现有系统工具观察 RAM；没有测到的数据不填估计值，也不为补齐表格建设监控系统。
- 合并、共享权重、卷积 LoRA 和 FSDP 混合参数是主要业务难点。以最短训练链定位问题，避免将检查本身扩展成独立工程。

## 8. 当前用法与验证边界

开关为 `worker.actor.model.lora.full_coverage=true`，且 `rank>0`。开启后显式枚举全部支持的层，忽略旧 `target_modules`/`exclude_modules`，不再排除视觉。默认仍为原全参数模式。

现有 ImageNet-R 10/20-task 脚本增加 `LORA_FULL_COVERAGE`、`LORA_RANK`、`LORA_ALPHA`、`ACTOR_LR` 和 `CLASS_ORDER_SEEDS`。下面是集群上首次验证的候选配置，不表示已调优；沿用已跑通的 4 卡、micro=1，其余数据与 RaPO 设置不变。先在实际环境确认依赖，再启动。

```bash
LORA_FULL_COVERAGE=true LORA_RANK=16 LORA_ALPHA=32 ACTOR_LR=1e-6 \
CLASS_ORDER_SEEDS=1990 NUM_GPUS=4 \
ACTOR_MICRO_BATCH_SIZE_UPDATE=1 ACTOR_MICRO_BATCH_SIZE_EXPERIENCE=1 \
bash scripts/img_cls/inr/train_10task.sh
```

结果位于 `saves/main_results/image_cls_cil/rapo_full_lora_r16/inr/10task/seed1990`，也可用现有 `SAVE_ROOT` 指定保存位置。完整 checkpoint 分片包含 LoRA 和直接训练向量，重载时使用相同 base model、rank/alpha 和 `full_coverage=true` 构建模型，再沿用现有 checkpoint 加载。它不是可直接传给 `AutoModel.from_pretrained()` 的合并 HF 模型；不输出会漏掉 norm 的 adapter-only 产物。

本地选择并检查了 `peft==0.18.0`（已提供 Conv3d LoRA），保留 `transformers==4.57.0` 的模型接口；集群若缺少相应能力，可在独立环境中安装该 PEFT 版本，不要求升级 vLLM。PEFT 对共享 embedding 的通用提示仍可能出现；本实现通过共享输出头和自身导出处理，不调用 PEFT 的原地 merge。

本地检查使用 Python 3.13、PyTorch 2.10.0 CPU、PEFT 0.18.0、Transformers 4.57.0，随机初始化微型 Qwen2-VL，无下载模型。检查了图文前向、全部可训练参数的梯度路径、冻结矩阵不变、合并数值、共享权重、模型状态重载和 Ref/Anchor 快照隔离。Windows Gloo 不可用，未完成 FSDP 运行检查；这些结果不代表作者 PyTorch 2.6/CUDA/vLLM 0.8.1 环境已验证。首次集群运行仍需确认 FSDP 包装/聚合、vLLM 装载、保存分片重载及 RAM/VRAM 峰值。

## 9. 参考依据

- 本地模型构建：[fsdp_workers.py](../../verl/workers/fsdp_workers.py)。
- 本地生成同步：[fsdp_vllm.py](../../verl/workers/sharding_manager/fsdp_vllm.py)。
- 本地任务切换：[image_cls_cil_rapo.py](../../examples/baselines/img_cls_cil/image_cls_cil_rapo.py)。
- [Qwen2-VL-2B 原始配置](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/config.json)。
- [PEFT LoRA 与权重共享说明](https://huggingface.co/docs/peft/main/package_reference/lora)。
- [vLLM 多模态 LoRA 支持](https://docs.vllm.ai/en/stable/features/lora/)、[adapter 加载限制](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py)。

外部文档于 2026-10-30 查询，`main`/`stable` 会变化。实现时记录实际选用版本，不把网页支持范围当作本项目端到端验证结果。
