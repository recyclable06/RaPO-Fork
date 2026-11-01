# 全参数覆盖的 LoRA 训练设计

更新日期：2026-11-01。状态：原版两卡运行完成 task1–4、96 次更新，在 task5 同步时 OOM。CPU 分片暂存、逐矩阵聚合与合并已实现，本地 FP32/BF16 数值检查已通过；新路径的 CUDA/FSDP/vLLM 运行仍待用户验证。

本文是本项工作的入口，保留已有模型设计并说明本次同步改造。[决策记录](decisions.md)记录取舍；[验证与实验计划](validation-plan.md)只说明本次改动的必要检查和后续实验建议。当前交付是一版可供用户在集群尝试的实现，不包含代跑完整实验或新建实验管理工具。

## 1. 要得到的结果

在现有 Qwen2-VL-2B-Instruct、ImageNet-R 类增量图像分类流程上，将全参数直接更新改为“矩阵及卷积核使用 LoRA，归一化和偏置向量直接训练”，使原来全部可训练参数的有效值都有更新途径。准确称为 **LoRA + norm/bias**，不称为每种参数都经过低秩分解的纯 LoRA。

保留同一套适配参数依次学习各任务、测试时判断全部已见类别的方式。保留 RaPO 的原始 Ref、上一任务 Anchor、Retention Reward 和 CTAN。新实验与已有全参数结果比较准确率、遗忘和资源消耗。

两组实验从同一原始模型版本开始。不得把已学完 ImageNet-R 的全参数 checkpoint 当作 LoRA 起点进行更新方式对照。

本次修改先解决已经发生的同步显存峰值问题。用户明确不设固定降幅、速度或 RAM 验收门槛；先实现一版，由用户在集群尝试运行，再依据已有监听数据分析峰值、耗时和 RAM，决定是否继续修改。降低峰值仍是目标，但不把规划估计写成保证，也不以未 OOM 代替完整的资源收益结论。

## 2. 已确认的范围与边界

- 用户要求覆盖全部参数，而非仅语言侧或仅 `all-linear`；向量采用直接训练。
- 模型的视觉、连接、语言部分均纳入覆盖检查；不因软件接口不支持而静默冻结某类参数。
- 实验室没有可直接复用的已验证实现；此前允许评估独立新环境。本次保留现有训练栈和普通 vLLM 权重加载，不以升级依赖作为前置步骤。
- 沿用现有仓库、训练入口、任务序列、日志和实验结果结构。完整实验及与全参数结果的比较是后续研究工作，不作为本次同步代码交付的前置条件。
- 本项不建设调度器、自动重试、监控服务、实验数据库、通用 adapter 路由或新分布式框架。
- 本次代码修改集中处理权重同步峰值，保留训练、生成和任务切换语义。实验超参数可按实际需要调整，不受此代码范围限制；解释资源变化时说明配置差异，不将其他配置带来的变化全部归因于同步改造。

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

沿用现有初始化覆盖检查和数量输出。本次不改参数注入与优化器范围，不重新建设参数审计或逐参数梯度检查；合并后的覆盖与共享关系属于新同步逻辑本身，仍需正确。

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

作者原有选择性 LoRA 路径通过 `disable_adapter()` 将 Actor 当作 Ref。直接训练 norm/bias 后，关闭 adapter 并不能恢复这些向量的初始值，因而不再代表原始策略。当前全覆盖实现已采用独立 Ref，复用全参数路径的角色划分，避免增加参数来回恢复的机制。

### 有效权重导出

- 使用参数名、形状和明确的共享关系对应权重，不使用两个模型 `parameters()` 的位置顺序配对。
- 导出不得改变训练端被冻结的矩阵，不永久合并或删除 Actor 的 adapter，不重置其优化器。
- 使用固定输入检查合并后的输出数值，考虑 BF16 舍入；这用于发现漏层或重复增量，不追求不同后端位级一致。
- 使用 FSDP 支持的分片状态接口；各 rank 的集体通信顺序必须一致。本次按单个权重矩阵聚合，并在矩阵内部按行分块计算 FP32 增量。
- 原实现先用 `full_state_dict=True` 汇总完整模型，再构造合并结果；这条路径已经发生 OOM。本次替换生产同步路径，不允许先汇总完整模型再套一个逐层循环。
- 每次同步更新完整有效策略，沿用现有按批次生成和缓存生命周期。仅在发现旧权重缓存被复用的具体问题时处理，不另建缓存管理或校验系统。
- vLLM 使用普通权重加载路径，不同时启用独立 LoRA 分支，防止同一增量被应用两次。

### 本次低峰值同步设计（代码已实现，待集群验证）

“逐层”实际指逐个权重矩阵，不是一次处理整个 Transformer block。保持现有单节点、FSDP、vLLM `TP=1` 布局，复用公共状态接口；不手工计算 FSDP flat parameter 偏移。

#### Actor 到 vLLM

1. 在 vLLM 仍休眠时，用 `get_model_state_dict` 和 `StateDictOptions(full_state_dict=False, cpu_offload=True)` 取得 Actor 的 CPU 分片快照。不读取优化器、不写磁盘。这里的 CPU 暂存只针对快照，不等于改变训练参数的 offload 配置。
2. 快照完成后唤醒 vLLM。保留现有唤醒接口；旧版若不能分别唤醒权重和 KV cache，不为此升级或修改 vLLM。
3. 各 rank 按相同顺序，取出当前权重所需的 `W0/A/B` 分片，移到计算设备，再用 `DTensor.full_tensor()` 聚合。CPU 分片不能直接通过 NCCL 做 CPU all-gather；不要在本地分片上直接计算 `B @ A`，其切分维度未必满足局部矩阵乘法语义。
4. 只生成当前矩阵的完整 BF16 有效权重；FP32 转换、乘法、缩放和加法按行分块执行，避免完整 FP32 `W0`、delta 和加法结果同时驻留。Embedding 按自身转置关系处理；Conv3d 展平后计算并恢复形状；norm/bias 直接使用当前值。
5. 将当前完整矩阵交给 vLLM 原有 `load_weights()`，沿用名称转换、QKV/gate-up 打包等现有加载逻辑。不要把同名矩阵的局部行反复交给普通 loader；分块发生在合并内部，加载输入仍满足完整形状要求。
6. 加载后释放当前完整输入、输出及临时量，再处理下一项。避免在循环外积累完整 GPU 权重字典，不逐矩阵调用 `empty_cache()` 或打印日志；保留现有阶段级缓存管理。

分布式聚合由固定的生产循环驱动，完成当前项的聚合后再调用 loader，避免 loader 的跳过权重行为改变通信序列。元数据使用全局形状和模型层定义，不从 FSDP 当前本地参数视图猜测形状。共享 embedding/head 使用同一有效权重，接收端允许跳过别名时也不能漏掉真正的更新。

#### Actor 到 Anchor

Anchor 是普通结构的 FSDP 模型，不照搬 vLLM 的逐项写入方式。当前实现：

1. 在 vLLM 休眠期间准备 Actor 分片快照，并从 Anchor 分片状态取得目标名称、全局形状和 DTensor 布局。
2. 复用上面的逐矩阵合并逻辑，将每个有效矩阵通过 DTensor 的分发/重分片接口转成 Anchor 目标布局，只在 CPU 字典保留本 rank 的分片。处理后的完整 GPU 矩阵立即释放；目标模板数值及时替换，避免保留不必要副本。
3. 全部参数齐备后，释放 Actor 快照，用一次 `set_model_state_dict(..., full_state_dict=False, strict=True)` 写入 Anchor。这里字典的“完整”指名称齐全，数值仍是分片。

这一接收方式已写入代码，仍待真实多卡验证。采用 `DTensor.from_local` 标记各 rank 已有的完整有效矩阵为复制布局，再用 `redistribute` 转成 Anchor 目标布局并转回 CPU。PEFT 名称清理、共享 embedding/head、Anchor 的 CPU offload 与目标分片布局是主要检查点。不按 Actor/Anchor 的 `parameters()` 位置配对，也不逐矩阵对整个 Anchor 反复执行 `load_state_dict(strict=False)`。

#### 资源和复杂度边界

- 分片状态导出内部仍可能展开一个 FSDP 单元，并产生临时副本；Anchor 加载也有展开开销。把这些阶段放在 vLLM 休眠时，不能据“最终结果是分片”宣称没有峰值。
- CPU 暂存以分片为单位，避免每个 rank 长期持有完整模型。实际 RAM、缓存和通信缓冲开销以监听数据为准，不把此前粗略预算写成预留门槛。
- 本路线降低同时驻留量，仍保留独立 vLLM 模型，并传输整模型量级的数据；CPU 往返及逐矩阵通信可能增加耗时，后续看实际影响。
- 首版内部合并块大小为 1024 行，不增加一组调参开关、双缓冲、异步传输或多种同步后端。
- 复用少量普通函数即可，不为两种接收端建立通用导出器类体系、后端注册表或参数登记框架。Anchor 的目标布局转换只服务当前模型和分片方式。
- 名称、设备、形状等局部适配由实现直接处理。只有证据表明需要深入修改 FSDP 私有结构、重写 vLLM loader 或迁移框架时，再讨论路线变化；不把每个报错都变成审批步骤。

## 5. 环境路线与实现位置

新版 vLLM 已增加 Qwen2-VL 视觉/连接 LoRA 支持，但其独立 adapter 路径仍有 bias、额外可训练模块等限制。升级不能单独满足本文的全参数覆盖目标。

本次选择保留已运行的训练栈与普通 vLLM 权重同步路径，采用 CPU 分片暂存和逐矩阵合并。新版原生 adapter 路线保留为后续选项，本次不同时实现，不开展多版本兼容矩阵或整体框架迁移。

本次代码落点：

| 位置 | 职责 |
|---|---|
| `verl/utils/full_lora.py` | 已有参数覆盖、共享输出头和完整合并；本次加入 CPU 分片快照与逐矩阵有效权重生成，供两个接收端复用 |
| `verl/workers/sharding_manager/fsdp_vllm.py` | 调整快照与唤醒顺序，逐项加载普通有效权重 |
| `examples/baselines/img_cls_cil/image_cls_cil_rapo.py` | 将有效权重转换为 Anchor 分片，一次写回 |

参数覆盖、独立 Ref、FSDP 包装、优化器、checkpoint 和启动入口沿用现有实现。实验超参数由现有配置选择，支持调参不等于需要开发调参工具。

这是一次局部但并非简单改参数的实现：主要难点是 Anchor 写回的布局对应，其次是分块合并的数值与共享语义。保留这些业务必要工作，不以“代码少”为由漏更新参数，也不将其扩展为通用同步框架。

## 6. 沿用保存路径

本次改变同步时的临时权重组织，不改变 Actor 参数结构或 checkpoint 格式。现有 checkpoint、CTAN/任务结果保存继续使用；没有相关改动或故障证据时，不为本次交付另做保存重载、HF 模型导出或恢复演练。

之后需要使用保存结果时再确认其可加载性；目前不宣称实际分片重载已验证。Anchor 的内存写回是任务切换业务，仍在本次范围内；磁盘恢复、回滚和断点续训不在本次范围内。

## 7. 实验配置与待测项

- 说明模型起点、数据与任务设置、评估口径和训练配置，使结果能够解释；不要求 LoRA 的训练超参数与全参数更新一致。
- 学习率、rank、alpha、batch、rollout 数量和训练轮数等均可试验调整。师兄建议学习率提高约 10 倍，可作为候选尝试，不是固定倍率或已验证最优值；不要求先沿用原学习率完成一轮。
- 卡数和 micro-batch 按可用资源与实际运行情况选择，已有可运行配置可作参考，不作为必须保持的要求。分析显存和耗时差异时说明这些配置变化。
- 参数与优化器开销预计减少，但完整权重同步、Ref/Anchor、激活和生成缓存仍存在。优先从已有日志取得耗时和显存，并用现有系统工具观察 RAM；没有测到的数据不填估计值，也不为补齐表格建设监控系统。
- 合并、共享权重、卷积 LoRA 和 FSDP 混合参数是主要业务难点。以最短训练链定位问题，避免将检查本身扩展成独立工程。

## 8. 当前用法与验证边界

开关为 `worker.actor.model.lora.full_coverage=true`，且 `rank>0`。开启后显式枚举全部支持的层，忽略旧 `target_modules`/`exclude_modules`，不再排除视觉。默认仍为原全参数模式。

现有 ImageNet-R 10/20-task 脚本增加 `LORA_FULL_COVERAGE`、`LORA_RANK`、`LORA_ALPHA`、`ACTOR_LR` 和 `CLASS_ORDER_SEEDS`。下面保留现有四卡启动示例，其中 `ACTOR_LR=1e-6` 仅表示原示例取值，不是推荐固定值；若尝试约 10 倍学习率，可改为 `1e-5`。卡数及其他超参数也可按本次实验调整。该示例未调优，也不表示本次分片方案已运行验证。

```bash
LORA_FULL_COVERAGE=true LORA_RANK=16 LORA_ALPHA=32 ACTOR_LR=1e-6 \
CLASS_ORDER_SEEDS=1990 NUM_GPUS=4 \
ACTOR_MICRO_BATCH_SIZE_UPDATE=1 ACTOR_MICRO_BATCH_SIZE_EXPERIENCE=1 \
bash scripts/img_cls/inr/train_10task.sh
```

结果位于 `saves/main_results/image_cls_cil/rapo_full_lora_r16/inr/10task/seed1990`，也可用现有 `SAVE_ROOT` 指定保存位置。完整 checkpoint 分片包含 LoRA 和直接训练向量，重载时使用相同 base model、rank/alpha 和 `full_coverage=true` 构建模型，再沿用现有 checkpoint 加载。它不是可直接传给 `AutoModel.from_pretrained()` 的合并 HF 模型；不输出会漏掉 norm 的 adapter-only 产物。

本地选择并检查了 `peft==0.18.0`（已提供 Conv3d LoRA），保留 `transformers==4.57.0` 的模型接口；集群若缺少相应能力，可在独立环境中安装该 PEFT 版本，不要求升级 vLLM。PEFT 对共享 embedding 的通用提示仍可能出现；本实现通过共享输出头和自身导出处理，不调用 PEFT 的原地 merge。

原实现的本地检查使用 Python 3.13、PyTorch 2.10.0 CPU、PEFT 0.18.0、Transformers 4.57.0，随机初始化微型 Qwen2-VL，无下载模型。检查了图文前向、全部可训练参数的梯度路径、冻结矩阵不变、合并数值、共享权重、模型状态重载和 Ref/Anchor 快照隔离。Windows Gloo 不可用，未完成本地 FSDP 运行检查。

2026-10-31 的两卡实际运行完成 task1–4、96 次更新，经过生成、训练、保存和任务切换，在 task5 首次同步的 FP32 合并表达式处申请 892 MiB 失败。这证明原链路已执行多个任务，不代表数值一致性、checkpoint 重载或完整十任务已验证。详见工作区诊断记录 `diagnostics/20261031-207-lora-2gpu/diagnosis.md`（位于 RaPO 仓库外的工作区根目录）。

本次分片改造由用户在集群尝试运行。关注 Anchor 分片写回及后续使用，以及“生成—更新—重新同步—生成”；通过实际运行覆盖一次任务切换即可，不要求先通过一套独立两卡预验证流程。必要的局部检查聚焦新合并与写回逻辑，具体见[验证计划](validation-plan.md)。性能与 RAM 根据已有监听数据分析，不预设门槛。

本次本地检查复用上述 CPU 环境：已有测试共 3 项通过，其中新增检查覆盖非零增量及更新后的向量、FP32/BF16 分块合并、非整除块边界、共享 head、Actor 不被合并修改，以及普通 CPU 张量的 Anchor 状态写回。生产同步已移除完整模型聚合；原 `effective_lora_state_dict` 仅保留为小模型数值对照。未运行真实 CUDA 聚合、DTensor 目标分片写回或 vLLM 实际装载，不将本地检查解释为显存降幅或多卡成功。

## 9. 参考依据

- 本地模型构建：[fsdp_workers.py](../../verl/workers/fsdp_workers.py)。
- 本地生成同步：[fsdp_vllm.py](../../verl/workers/sharding_manager/fsdp_vllm.py)。
- 本地任务切换：[image_cls_cil_rapo.py](../../examples/baselines/img_cls_cil/image_cls_cil_rapo.py)。
- [Qwen2-VL-2B 原始配置](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/config.json)。
- [PEFT LoRA 与权重共享说明](https://huggingface.co/docs/peft/main/package_reference/lora)。
- [vLLM 多模态 LoRA 支持](https://docs.vllm.ai/en/stable/features/lora/)、[adapter 加载限制](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py)。

本次分片路线的源码依据（2026-11-01 调研）：

- [PyTorch 2.6 状态接口](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/checkpoint/state_dict.py)：分片状态和 CPU offload 选项。
- [FSDP 状态导出与加载](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/fsdp/_state_dict_utils.py)：分片接口内部仍有模块展开和聚合。
- [DTensor 接口](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/distributed/tensor/_api.py)：完整张量聚合与分发。
- [vLLM 0.8.1 加载器](https://github.com/vllm-project/vllm/blob/v0.8.1/vllm/model_executor/models/utils.py)、[embedding 加载](https://github.com/vllm-project/vllm/blob/v0.8.1/vllm/model_executor/layers/vocab_parallel_embedding.py)：逐项加载与完整形状要求。

上述接口依据不等于本项目组合已验证；`main`/`stable` 链接会变化，实际行为以运行版本为准。
