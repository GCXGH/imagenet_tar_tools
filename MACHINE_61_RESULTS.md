# 61机器实测与可迁移结论（2026-11-09）

## 摘要

在本机已测布局中，选择**原字节TAR、两组各4卡并行、每rank4 workers**。
历史ImageFolder每组约60–61分钟/轮，TAR完整遍历诊断约12.30分钟/轮，
整体速度约为原来的4.9倍、时间减少约80%。这是历史正式训练与隔离诊断的比较，
不是严格控制缓存后的纯格式因果结论，也不是其他设备可直接承诺的收益。

## 1. 环境与负载

- 8张RTX5090，Dense约60M参数、MoE4约175M参数，均为3层、约11.8GMAC/image。
- 数据位于`/mnt/data`，系统将其报告为14.6T、ROTA=1、MR9361-8i RAID逻辑盘；
  不据此推断每块物理盘规格。GPU拓扑为SYS互连。
- torch2.9.0+cu130、torchvision0.24.0+cu130、timm0.4.12。
- 有效batch1024、BF16；4卡batch256/卡，8卡batch128/卡，accumulation1。
- 数据train1,281,167、val50,000，4096样本/片，313train＋13val shards。
- 完整TAR校验和回归测试通过；训练每轮1251 updates，略去143train尾样本。
- 冻结来源为原训练第22轮checkpoint（内部epoch21，update27522），诊断恢复AdamW，
  固定该阶段LR；真实反向与更新仅在内存，不写checkpoint，原输入哈希前后一致。
- 缓存未控制、prefetch1、每step CUDA同步；这些条件均限制对正式训练的外推。

## 2. 同worker后端比较与worker扫描

以下为两组各4卡并行的短测，取较慢组：

| 输入 | workers/rank | 秒/update | 比较说明 |
|---|---:|---:|---|
| ImageFolder | 1 | 5.164 | 同配置基线 |
| TAR | 1 | 1.522 | 同workers约3.4倍速度，时间减少70.5% |
| TAR | 2 | 0.777 | 本机增加workers有收益 |
| TAR | 4 | 0.441 | 首次短测 |
| TAR | 4 | 0.482 | 重复短测 |
| TAR | 4 | 0.549 | 完整遍历测量段，不能把短测最佳当长期速度 |

短测warmup10＋测量60；完整遍历warmup10＋测量1241，所有共1251更新的图像均经过输入流程。
不能拿ImageFolder workers1与TAR workers4直接声称“纯格式加速11.7倍”。

## 3. 完整遍历与调度

| 布局 | Dense秒/update | MoE秒/update | 两组各推进一轮的估算时间 |
|---|---:|---:|---:|
| TAR 4+4同时运行、w4 | 0.549 | 0.549 | 12.30分钟 |
| TAR 各8卡顺序运行、w4 | 0.443 | 0.467 | 20.13分钟 |

整轮估算按测量段秒/update×1251，再加实测完整验证；含验证、不含checkpoint保存和重启。
8卡Dense每轮约9.95分钟，MoE约10.18分钟；要加起来才是顺序完成两组的成本。
4+4总完工比8卡顺序少约39%，两组进度吞吐约1.64倍；该收益不能再乘到4.9倍上。

GPU驻留合成输入：Dense4→8卡0.148→0.091秒/update，MoE0.212→0.197。
MoE扩卡收益明显更小，与互连/通信成本相符，但没有单独profile证明唯一瓶颈。

## 4. 历史正式训练与100轮时间外推

原ImageFolder Dense/MoE末轮分别3617.83/3612.72秒，末5轮均值3663.25/3662.65秒。
约60.2分钟→12.3分钟，对应约4.9倍速度、时间减少79.6%、每轮节约约48分钟。

| 方案 | 两组并行各100轮的墙钟估计 |
|---|---:|
| 历史ImageFolder速度 | 约100–102小时（4.2天） |
| TAR完整遍历诊断速度 | 约20.5小时，未计checkpoint保存等 |

这是同100轮预算的外推，不是说原模型实际完成100轮；旧运行在22轮完整保存后停止。
61机器后来用户启动的新TAR＋QK norm实验早期约0.55–0.58秒/update，
加假设65秒验证/保存后约12.8分钟/轮、100轮约21.3小时（约4.7倍，节约约79–80小时）。
**这只是2026-11-09 19:26的首轮未完成快照，不能当已测100轮或TAR纯加速证据。**
此处保留它以区分实测和ETA，不把随时间失效的预计结束日期当通用结论。

## 5. 遥测线索

10秒粒度、包含启动/warmup/验证的辅助均值：ImageFolder workers1读盘约24.8MiB/s、
逻辑盘busy95.6%、8卡平均利用率21.4%；TAR 4+4完整遍历约173.3MiB/s、86.4%、67.6%。
这些支持小文件访问成本减少，但不是严格匹配的纯训练遥测窗口。
4+4整轮还观察到约1420 swap-in页/秒平均值，说明内存/换页仍值得关注；没有清缓存或关闭swap。

## 6. 证据与复核

本库附[evidence/machine61_pipeline_summary.csv](evidence/machine61_pipeline_summary.csv)：
从原报告按显示精度转录的26项数值摘要，非原始高精度JSON，不含图片或checkpoint。

原机证据根：
`/mnt/data/xinghaitong/results/imagenet_tar_validation_v1/matrix_epoch22_v2/`

| 原文件 | SHA256 |
|---|---|
| `REPORT.md` | `8e0475da0aa70ab13af8bb1c4c3a671cc058c9dc821723c8aed5f495cebdecce` |
| `results.json` | `fe0a7d0a695cf1de908fe81a973e05d13b844faab4b2787929c9292d2e2f93c7` |

原目录另保留命令、日志、遥测、硬件信息、源码快照、失败项（空）和checkpoint不变记录。
原ImageFolder指标文件位于`/mnt/data/xinghaitong/results/compute_matched_vit_v1/`下，
模型名`dense_3layer_11p8g`/`moe4_top1_3layer_11p8g`，run均为
`seed0_main_61_4gpu256_imagefolder_v2/metrics.jsonl`：

- Dense SHA256：`81d5e9ac5481252f84f9dc0b428e355d629aede91a645ebafe6d663d61421541`
- MoE SHA256：`f3c25a1d66441a372c6626f9dbfd8226bd82bd1f151d9a260e46ea07f3afbc59`
- 数据manifest SHA256：`108a8f1bbc2b47434366f99752e02fc71ebee7020a20ff4363851957a135716e`

这些绝对路径仅用于原机追溯，不要求其他机器存在；完整原始证据未复制到本库。
跨设备应保存自己的原始结果。未测workers8/NUMA/更多buffer策略，不声称本机全局最优。
