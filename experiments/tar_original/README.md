# TAR original v1：冻结的生产基线

## 已有实现，不重打包

- 转换器：本仓库根目录 `prepare.py`、`verify.py`，commit
  `4c3ffd5968d1ab3fd926fa77cbdf2835f8f5e85f`。
- `prepare.py` SHA256：`3a03933cbf1df9b6f75e6900bb3275d5a151245ea50dee128615e929c9224924`。
- 图片原字节写入未压缩 POSIX USTAR；image 与 `.cls` 成对，类别按 ImageFolder 排序。
  转换时先混合类别；JSONL 保存 source/key/label/hash，manifest 最后原子发布。
  分片级恢复会核对转换代码、设置、源元数据、archive/sidecar hash。
- 数据：`/data1/xinghaitong/dataset/imagenet1k_original_tar_v1/`。
  train 1,281,167 张 / 313 shards；val 50,000 张 / 13 shards；4096 samples/shard。
  archives train 149,204,930,560 bytes，val 6,796,175,360 bytes。
- manifest SHA256：`f855f53ee416fdbb27e32b1102365235cbf9a3bad6efe57fd9c46974478308ef`。
  2026-11-09 约20:01完成转换及全量 payload/label/唯一性/计数校验，有匹配的 verified.json。
  本轮未重新遍历全数据做 hash。

生产 reader：`/data1/xinghaitong/projects/spb_vit_dist/ablation/pbi_width_scaling/tar_data.py`。
本次读取时 SHA256：`54f598bff289fb7235f41677e1a32a44eaeaa9ea4999df70b0ef0084bf6066fb`。
它是独立 IterableDataset，不使用 WebDataset 库。不要将该结果称为 WebDataset 库的 benchmark。

每 epoch shard shuffle + circular rotation → disjoint rank ranges → batch-aligned worker ranges。
4MiB Python 文件缓冲、tarfile `r:`，每段开头扫描成员头并跳过 prefix，主体顺序读 payload。
每 worker 压缩字节 shuffle512，再 PIL RGB + 原训练增强。并非全局 ImageFolder permutation。
验证精确覆盖一次；训练 batch1024 每轮1251 updates，保留1,281,024张，丢143张全局尾部。

正式设置：物理 GPU4–7，4×256×accumulation1，BF16，compile default，
8 workers/rank，prefetch1，pin_memory=false；Dense/PBI保持一致。
pin memory 曾在一次短测中报 CUDA invalid argument，重试/隔离检查未复现，根因未定。
当前关闭它是保守配置，不代表所有机器都应关闭。

## 已完成验证和性能

历史回归：15项训练/模型测试及6项 TAR 测试通过，包括多 rank/worker 计划、
字节/标签、epoch确定性、两 rank Gloo训练/验证/checkpoint/restore。
宽模型四卡短测梯度和 eager 验证有限。详见 [results.csv](results.csv)。

历史全量四卡短测物理 GPU1–4：Dense ImageFolder约3.28–3.37秒/步，
TAR4 workers约0.568–0.617秒/步，TAR8 workers较长200更新平均0.4613；
PBI TAR8 workers平均0.5273。样本顺序/缓存不同，不是严格同字节冷缓存 TAR-only 加速。
原 JSON/log：`/tmp/opencode/imagenet_tar_full_bench/`。
小pilot同顺序 CPU增强 files/TAR 仅约1.5–3.3%差异，不能外推 HDD 全量。

### 新增证据：完整 PBI wider 100 epochs

结果目录：
`/data1/xinghaitong/results/spb_vit_experiments/ablation/pbi_width_scaling_v1_scripts4/pbi_wide_100e/seed0_main_tar_compile_4gpu_b1024/`。

2026-11-09 21:25开始，2026-11-10 15:51结束。metrics有100条，
总125,100 updates，所有轮 val_samples=50,000，AMP skips合计0。
最终Top1=79.288%，最佳79.328%（第98轮）。证明该配置完成了此次长跑，
不证明比 ImageFolder 精度相等，也不代表 Dense/300e 已经验证。

排除首轮（含编译）后的99轮均值：

| 指标 | 值 |
|---|---:|
| train | 618.569秒/轮 |
| train / 1251 | 0.494459秒/更新 |
| 逐轮吞吐均值 | 2071.53 images/s |
| validation | 40.024秒/轮 |
| checkpoint | 2.851秒/轮 |
| 完整 epoch | 661.476秒（11.025分钟） |
| rank0 data wait | 79.301秒/轮，累计比12.82% |
| 全100轮 epoch_seconds合计 | 18.426小时 |

rank0 data wait只量 next batch 的可见等待，不含所有解码/传输或其他rank等待。
消除它也不意味着自动获得合成数据0.289秒/步；必须剖析H2D、增强、同步和rank差异。
本次长跑使用GPU4–7；不能与旧GPU1–4短测作严格方案A/B对照。

原 metrics.jsonl SHA256：`f5f6e2b4430b9ca18024ffc6db7c8b6fff910c4d5aee74f278a282173d275433`。
现有 trainer SHA256：`550bbf042fab66bba6291bd70bfcb64574cccf5a8be6b4b5128f352d34bdf8ee`。
本轮仅读取生产文件，不更改训练源码、恢复签名或结果。
