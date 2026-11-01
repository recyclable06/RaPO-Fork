# ImageNet-R LoRA 10-task 启动说明

2026-10-31 已在 207 确认 `/home` 为 rw，服务器代码为 `a75ab9c`。模型、数据、原 Conda 环境路径可读。空闲 GPU 和 RAM 需要在实际运行前重新确认。

## 默认设置

启动器：`scripts/run_inr_lora_local.sh`。它调用现有 `scripts/img_cls/inr/train_10task.sh`，不修改训练实现。

| 项目 | 默认值 |
|---|---|
| GPU | 命令行指定两张或四张不同物理卡，NUM_GPUS 自动取卡号数量 |
| 模型 | `/home/zhenglifeng/models/Qwen2-VL-2B-Instruct-895c3a4` |
| 数据 | 仓库内 `data/image_cls_cil_dataset/imagenet-r`，使用 train_5shots 和 test |
| 结果 | `/mnt/conda/zhenglifeng/rapo-fork/inr-10task-lora-年月日-时分秒` |
| 全覆盖 LoRA | true；矩阵参数使用 LoRA，norm/bias 等向量参数直接训练 |
| rank / alpha | 16 / 32 |
| 学习率 | 1e-6 |
| 类别顺序 | 只跑 1990；完整 10-task |
| update / experience micro-batch | 1 / 1 |
| 其余训练参数 | 沿用底层脚本默认值，包括 2 epochs、global batch=8、rollout n=8 |

可用环境变量显式覆盖表中 rank、alpha、学习率、种子、micro-batch 及模型/数据/保存路径。全覆盖开关固定为 true；GPU 数由卡号自动确定。首次不建议同时尝试多组配置。

## 启动顺序

本机 PowerShell 中登录已安排资源的节点，例如：

```powershell
ssh 207
```

以下均在该节点的 Linux 终端执行。新建 tmux；如已在其中，不必再建或嵌套：

```bash
tmux new -s rapo-lora
source /home/zhenglifeng/miniforge3/etc/profile.d/conda.sh
conda activate /mnt/conda/zhenglifeng/rapo-author-diagnostic-20260908/env/rapo-author
cd /home/zhenglifeng/projects/RaPO-Fork
nvidia-smi
free -h
cat /proc/pressure/memory
```

确认所用 GPU 的进程和使用安排；GPU 利用率为 0 不等于无人使用。RAM 查看 available；PSI full 持续升高表示严重等待内存。还需注意整台节点的 user.slice 限制，资源日志会保存相关值。现有 CPU 测试不能预测真实 LoRA 分布式训练的 RAM/显存峰值。

假定已经确认可用的是物理 2、3、4、5 号卡：

```bash
bash scripts/run_inr_lora_local.sh 2,3,4,5
```

将这四个数字替换为实际安排使用的卡。若安排使用的是两张卡，例如 4、5，则执行：

```bash
bash scripts/run_inr_lora_local.sh 4,5
```

两卡和四卡均无需另外 export GPU 数量、模型、数据和 LoRA 参数，也无需再加一层 tee。CUDA_VISIBLE_DEVICES 只是限制可见设备，不负责预留卡。

启动器先打印本次结果目录和可直接复制的监视命令，然后逐卡执行小张量 CUDA 检查；全部成功后进入训练。这不是预先占满显存或预留 GPU。

## 监视与保存

启动器自动每 5 秒采样并写入本次目录的 `resources.log`：仅所选 GPU 的显存与利用率、整机 RAM/Swap、整机 user.slice 及本账号 slice 的 current/high/max、整机内存 PSI。

在 tmux 中按 Ctrl+b，松开后按 %，打开新窗格，粘贴脚本打印的 `tail -n 40 -f .../resources.log` 完整命令。此窗格不用激活 Conda。Ctrl+C 仅退出 tail 查看；训练窗格继续运行。Ctrl+b 后按方向键切换窗格；Ctrl+b 后按 d 可离开 tmux，回到同节点后执行 `tmux attach -t rapo-lora` 查看。

- `free -h` 的 available 是整机可用内存，不是实验占用。
- `memory.current/high/max` 单位为字节（除以 1024³ 得到 GiB），`max` 表示无该项有限额；user.slice 是所有相关用户的合计，不是本实验单独占用。
- PSI `full avg10` 是最近十秒内存完全阻塞时间的百分比。
- 5 秒采样可能遗漏瞬时峰值；LoRA 仍需加载基模、参考模型、anchor 和 rollout 权重，不保证只占少量显存或 RAM。

训练脚本正常或异常退出时，本启动器会结束自己启动的监听进程，不会停止别人的监视或训练。

| 保存文件 | 用途 |
|---|---|
| console.log | 底层训练脚本的标准输出和报错、最终退出码 |
| cuda-check.log | 启动时所选 GPU 的 CUDA 小张量检查输出 |
| resources.log | GPU、RAM、Swap、cgroup 和 PSI 采样 |
| run-info.txt | 节点、Python、commit、启动设置和训练退出码 |
| gpu-before.csv / launch.sh | 绑卡快照、启动脚本副本 |

资源监听本身不占 GPU 显存；它读取 nvidia-smi 和系统信息。控制台日志不包括随后在其他窗格执行的命令输出。

## 完成判据与已做验证

首轮成功应同时看到退出码 0，以及 `image_cls_cil/rapo_full_lora_r16/inr/10task/seed1990/<时间>/cil_info/metrics/summary.json` 中的 task1–task10。这证明一个 LoRA 种子完成，不等于三个种子或论文全部实验完成。

2026-10-31 在原服务器环境执行了已有 `test_full_lora.py`：2 个 CPU 小模型测试通过，覆盖参数更新、权重导出、anchor、保存重载及未覆盖参数报错。另通过启动器 Bash 语法检查和模拟训练下的默认传参、退出码 0/7、日志及重复 GPU 拒绝检查。模拟检查在临时目录进行并已清理，未调用真实 CUDA 或启动实际训练。

启动器随后扩展为支持两卡或四卡，并模拟检查两种卡数的传参与退出码。上述验证不包含真实两卡/四卡 FSDP、vLLM 同步与完整训练；这些需要首轮实际运行确认。

## 两卡试跑的判断依据

当前默认 rollout batch=8、global batch×rollout n=8×8=64、TP=1、Ulysses=1，均兼容两卡。四卡时每卡负责 16 条训练轨迹，两卡为 32 条；global batch 和每次更新的数据规模保持不变，micro-batch=1 下通过更多分组完成。

LoRA 主要减少可训练参数及其优化器状态；冻结权重、激活、独立 Ref/Anchor、vLLM 和完整有效权重同步仍有开销。两卡也会增加每卡的 FSDP 分片，不能由“使用 LoRA”推断两卡一定够用。尤其需要观察首次 vLLM 权重同步、首次更新、task1 保存和 task2 Anchor 更新，完整十任务结果才能确认单种子跑通。

先保持 TP=1、rank=16、alpha=32、global batch=8、rollout n=8、两个 micro-batch=1，不同时改动算法和 batch。若初始化或同步阶段 OOM，继续降低训练 micro-batch 通常不能解决该阶段的分配，需要根据具体栈决定下一步。
