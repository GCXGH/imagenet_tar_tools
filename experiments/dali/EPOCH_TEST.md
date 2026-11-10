# 四卡 batch1024 一轮对照（2026-11-10，用户明确授权）

用户要求扩大规模，允许完整epoch并使用四张空闲GPU。已确认GPU4–7空闲。
顺序运行 original / pil_matched / dali，每项1251次更新，batch4×256=1024，
PBI wider BF16 + compile default + AdamW + Mixup/CutMix + clip5，只有benchmark输出，
无验证/checkpoint，无正式训练启动。预计30–60分钟；每项硬超时30分钟，失败停止不重试。
不下载/复制数据，不drop_caches，不更改生产训练仓库。

## 为什么允许一个显式生产适配器

数据管线本体仍独立，不跨方案导入。此次`epoch_benchmark.py`的benchmark适配器
显式只读导入用户指定生产模型/训练step/原TAR loader，以保留原配方作参考，
而不是将模型复制或硬编码到通用库。生产源码hash保存到结果，不修改恢复签名。
这不是通用loader依赖生产repo：独立DALI管线/索引/reader检查均不需要它。

## 三组定义

- `original`：原TAR IterableDataset、timm rand-m9-mstd0.5-inc1、原pixel erase .25、
  workers8/rank、prefetch1、pinning false、shuffle512。
- `pil_matched`：同原TAR读取和分配，torchvision RRC224 bicubic + flip +
  RandAugment(2,9,bilinear,fill128) + Normalize，GPU pixel erase .25。
- `dali`：原生readers.webdataset、全局shard-order shuffle + sample buffer512、
  num_shards4、last_batch DROP；mixed decoder image_random_crop + GPU bicubic resize +
  flip + DALI RA(n2,m9,linear,fill128,monotonic) + Normalize，GPU pixel erase .25。

PIL/DALI同一增强族，不是完全同配方：RA算子集/幅度映射、RRC/resize/插值/取整、
随机数与采样顺序不同。因此性能对照不能证明原精度保留；original单独列作生产参照。
两组都会做RandomErasing、Mixup/CutMix和完整反向/AdamW，不能省略增强计算伪造加速。
原始标签是ASCII .cls，在DALI图中按1/2/3位十进制解析，而非错误cast原字节。

## 小型元数据索引

DALI需读取index。通过v1固定USTAR布局+已有JSONL累计offset，仅扫描元数据，
不读写图片payload，不重打包全量数据。每个样本写image/cls ext/offset/size/filename。
首次按文档展示省略filename被DALI2.4拒绝，保留诊断目录`imagenet_dali_indexes_v1`（不可用），
修正后另建`/data1/xinghaitong/dataset/imagenet_dali_indexes_v1_2`，256张原生reader
hash与label检查通过。不会覆盖或自动删除初版诊断产物。

## 预检

四卡12步完整训练通过，原生reader报告1251步/rank。各rank梯度全部存在且有限。
首步包含compile约32.5秒，post-first约.338秒/更新；这是短测，不是完整epoch结果。
预检发现DALI RandAugment要求enable_conditionals=True，已在独立代码修正。

## 监控和局限

每2秒记录GPU utilization/memory/power、MemAvailable/Cached及sda/sdb diskstats。
记录各rank acquire/H2D可见时间、p50/p95/p99、窗口、首步与post-first步时。
吞吐按最慢rank；compile首步计入epoch并另列。torch allocator峰值不含DALI显存，
总显存使用监控nvidia-smi。无逐step barrier，避免人为改变正常重叠。
按DALI reader meta+DROP计算计数；还未进行完整epoch sample ID逐一去重检查，
不把标签直方图计数当作唯一性证明。原生reader首批payload/hash校验与分片计数是已有验证。
全局1251×1024=1,281,024训练样本，143尾部不参与更新。
输出顺序和缓存状态不同，不是严格同字节冷缓存控制实验。单次顺序测试可能有缓存/负载偏差。

命令入口：`run_epoch_suite.py`，默认dry-run；--execute才启动三个串行进程。
批准的命令结果根：`/data1/xinghaitong/results/imagenet_io_benchmarks/dali_epoch_20261110/`。
所有case进程均有独立process group，只在超时/失败时终止自己的进程，不碰他人任务。
