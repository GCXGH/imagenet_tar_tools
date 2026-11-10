# 四卡全局batch1024：完整epoch结果

2026-11-10，17:24–17:54，物理GPU4–7，三个case串行完成，无后续自动任务。
完整定义见 [EPOCH_TEST.md](EPOCH_TEST.md)，不将改造配方称作原配方精度等价。

## 共同条件与正确性

- 同一PBI wider、4×256、BF16、torch.compile default、AdamW、反向、clip5、Mixup/CutMix。
- 每项1251次更新，共1,281,024个训练输入；各rank320,256个，143张尾部不参与更新。
- 所有rank最终参数梯度存在且有限，日志无OOM/异常退出。
- DALI存在内置RandAugment使用deprecated算子的警告，不影响本次完成，但后续需跟踪版本。
- 不含验证/checkpoint；epoch计时包含首步编译/数据启动，模型/pipeline setup在epoch外单列。
- DALI的全局sample ID唯一性未逐一审计，不能用更新次数或标签数替代该验证。

## 核心结果

| 管线 | epoch训练秒数 | images/s（含首步） | 首步秒数约 | 首步后秒/步 | setup秒数rank0 |
|---|---:|---:|---:|---:|---:|
| 原TAR + 原timm增强 | 624.94 | 2049.83 | 53.8 | 0.45692 | 2.15 |
| 原TAR + PIL对应增强族 | 623.42 | 2054.82 | 35.1 | 0.47062 | 2.27 |
| DALI原生TAR + mixed/GPU增强 | 458.99 | 2790.98 | 30.7 | 0.34264 | 8.94 |

DALI相对原TAR整轮吞吐+36.16%，训练时间减少26.56%；排除首步，吞吐+33.35%。
相对PIL增强族对照，整轮吞吐+35.83%，排除首步+37.35%。
原TAR与PIL对照整轮相近，但不代表两个增强配方计算量严格相等。
不同case首步编译差异会影响整轮比值，所以同时列post-first，不只报告最有利数字。

去掉首个50步窗口后的多数窗口：原TAR约0.435–0.481秒/步；
DALI约0.325–0.368秒/步，另有一个窗口0.420秒/步。收益持续到epoch末尾，
并非只在小子集/启动几步有效，也没有明显持续性退化。

## GPU/内存/数据获取

每2秒nvidia-smi采样，包含启动/编译和结束阶段，采样峰值不是严格峰值：

| 管线 | GPU4–7平均util | 单卡总显存最大采样 | 最低MemAvailable |
|---|---:|---:|---:|
| 原TAR | 67.2% | 19.95GiB | 50.68GiB |
| PIL对照 | 65.4% | 19.95GiB | 51.23GiB |
| DALI | 82.6% | 21.69GiB | 61.49GiB |

DALI比原TAR额外约1.75GiB总显存，但torch allocator峰值反而略低，
说明只看torch max_memory_allocated会漏掉DALI占用。当前配置单卡24GiB内存余量更小。
GPU util是kernel活动时间指标，不是SM/Tensor Core占用或峰值算力效率。

每rank一轮data acquire累计时间：
- 原TAR：180.4 / 194.0 / 203.3 / 190.3秒。
- PIL对照：191.7 / 166.0 / 184.9 / 174.0秒。
- DALI：10.8 / 7.6 / 16.1 / 8.4秒。

这里PIL路径含host-to-device，DALI输出已在GPU；都可能与GPU计算重叠，
不能直接称作纯磁盘等待或全部可节省时间。它也与旧正式日志只计next(loader)的12.8%不是同口径。

## 磁盘观测及归因限制

整个case期间sdb设备（含其他作业、setup/compile）的累计数据：

| 管线 | 设备读取GiB | 平均MiB/s | busy | 平均read await |
|---|---:|---:|---:|---:|
| 原TAR | 139.29 | 223.18 | 79.6% | 5.36ms |
| PIL对照 | 130.16 | 208.72 | 79.5% | 4.96ms |
| DALI | 120.29 | 253.97 | 79.2% | 2.37ms |

可见磁盘仍有大量工作，但不能仅凭busy约80%就断言物理盘已饱和。
这些是全设备共享统计，不是本进程归因；顺序固定、缓存不可控，后运行DALI可能获益于前序缓存。
DALI实际读字节较少不能全归为reader高效，也不能用这些数据量反推精确cache命中率。
要分离纯I/O、解码、增强与调度收益，需要后续同采样路径的组件消融和反序重复。

## 结论与后续

**原TAR之后仍有显著端到端加速空间；DALI是已验证可运行且值得继续投入的候选。**
这次同时改变reader、并发/缓存访问模式、decoder、增强执行位置和实现，
证明的是整套改造的性能，不是DALI reader或GPU decode单独贡献。
不能声称原TAR已经消除I/O，也不能声称上述33–37%来自纯磁盘优化。

现阶段不替换已经完成/在跑的正式实验，DALI仍是独立实验方案。
下一步优先：
1. 反向顺序重复至少一次，检查缓存和背景负载的影响。
2. 记录全epoch ID/rank唯一性与覆盖，检查多epoch/restart的采样语义。
3. 在DALI内做CPU/mixed decode的组件消融，进一步定位收益，不反复重打包。
4. 对齐/记录RA、RRC和插值差异，另行批准精度对照；性能结果不能代替收敛验证。
以上未自动启动，本次授权的三个case已经全部结束。

原始结果：`/data1/xinghaitong/results/imagenet_io_benchmarks/dali_epoch_20261110/`，
含case JSON、log、每2秒telemetry和summary。read-only分析脚本`analyze_epoch_suite.py`。
