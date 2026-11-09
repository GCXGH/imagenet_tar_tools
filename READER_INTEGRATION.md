# TAR训练读取器集成契约

本库不提供torch reader；下面是61机器已验证实现中可以借鉴的设计。
目标项目可选择WebDataset/DALI或自实现，但须测试其实际采样、错误处理和恢复语义。

## 1. 格式与完整性

- 完整manifest格式为`imagenet-original-webdataset-v1`；检查`complete`、类别索引连续性、
  `class_to_idx.json`一致性、split计数、分片路径位于数据根内且无重复、文件大小与清单一致。
- 完整`verify.py`核验图片payload、标签、唯一ID/来源、sidecar和分片哈希，
  但不解码图片，也不能代替DDP与增强测试。轻量启动检查不等于完整哈希核验。
- TAR为无压缩USTAR；每个样本是相邻、同basename的图片member和整数`.cls`member。
  顺序流读取，不解包到磁盘；检查成员类型、配对、允许扩展名、标签范围和截断。
- 原参考以1MiB文件缓冲、`tarfile`的`r|`流模式读取。边界处可能重新扫描分片前缀，
  不代表样本重复输出；需要用样本key测试输出覆盖。
- 损坏数据应报错，不静默跳过/补样，否则计数、类别分布与rank步数可能悄悄改变。

## 2. 有限epoch、DDP与worker分配

定义N为train样本数，W为world_size，B为每卡microbatch，A为梯度累积次数：

```text
effective_batch = W * B * A
updates_per_epoch = floor(N / effective_batch)
microbatches_per_rank = updates_per_epoch * A
used_train_samples = updates_per_epoch * effective_batch
omitted_train_samples = N - used_train_samples
```

原参考每轮先用`Random(seed + epoch)`全局打乱分片，然后按逻辑样本流给各rank分配
连续互不重叠区间；再按**完整microbatch数量**分配worker区间。worker预先拼batch，
DataLoader设`batch_size=None`，不额外自动拼batch。

这样不会因每个worker独立`drop_last`再丢一份尾部，也保证所有rank相同更新数。
不能给IterableDataset再套DistributedSampler而重复分片。通用按分片取模分配也能实现，
但分片长度和尾部不齐时，要另外证明各rank有相同步数，不会DDP挂起。

ImageNet例：N=1,281,167、W=4、B=256、A=1 →1251 updates、每轮使用1,281,024张、略去143张。
worker数不应改变这个计数。分片shuffle会改变尾部位置，但不是全数据的均匀随机排列，
不能声称每张图被遗漏的概率严格相等。

验证集：全量样本按rank/worker互不重叠分配、保留末尾不满batch、不shuffle、不pad、
无独立worker丢尾。ImageNet应恰好50,000张。各rank验证batch数量不一定相同，
避免每batch必须同步的训练式DDP forward路径；指标按全局样本总数归约，不能简单平均rank准确率。

## 3. 先shuffle压缩字节，再解码

- worker先读取原始图片payload和标签，放入有界shuffle pool，随机输出后才PIL解码/增强。
- 61机器采用每worker最多1024样本、128MiB**编码字节**的双限制；单个超大样本直接输出，
  不能无限涨buffer或丢样。Python对象额外内存、解码图片、batch张量、预取/共享内存另计。
- 8rank×4worker×128MiB，仅编码buffer上限之和就约4GiB，绝不是训练总CPU内存上限。
- 不在buffer中存已解码全分辨率图片；这会显著放大内存压力。
- PIL读取后转RGB，与原ImageFolder loader保持一致；之后原样使用训练/验证transform，
  Mixup/CutMix等仍在原位置执行。TAR没有省掉JPEG解码成本。

## 4. 进程、epoch和RNG

- 原参考设置`prefetch_factor=1`、pin_memory用于GPU输入、多worker使用`spawn`，
  避免CUDA/NCCL初始化后fork。启动main须保护入口，transform等对象须可pickle。
- worker非persistent：每次构造iterator能拿到主进程更新的epoch。
  若改persistent_workers，必须显式传播epoch，不能只调主进程dataset.set_epoch。
- 示例种子：`seed + 1_000_003*epoch + 1009*rank + worker_id`，分别初始化Python/NumPy/torch；
  shuffle用独立Random实例，避免buffer抽样消耗增强的全局随机流。
- 固定布局可复现不等于ImageFolder逐位等价。改变rank数、worker数、buffer、
  shard排序或后端会改变样本/增强轨迹。不要把这些变化藏在同一个正式run里。

## 5. 正式恢复和可追溯性

记录：数据manifest和reader源码SHA256、converter版本、world/batch/accumulation、workers、
buffer/prefetch、seed、epoch、transform、库版本、命令、模型和优化配方、源码快照。
原参考只承诺epoch边界恢复；没有流游标和全部worker RNG/buffer状态，不承诺任意step精确恢复。

ImageFolder→TAR应新建run、记录来源与变更，按训练系统的恢复规则审核模型/优化器/LR/RNG。
不绕过配方/源码签名；不因为张量可load就宣称等价恢复。
TAR只改变输入效率，不自动解决注意力饱和、LR不稳定等模型收敛问题。

## 6. 最低回归测试

1. 原始字节与标签逐项匹配；代表性图片RGB解码与固定seed transform输出一致。
2. train/val多rank、多worker key集合无重叠，符合上述全局计数；覆盖小数据、短分片、不可整除。
3. 相同seed/epoch/layout可复现；改epoch改变train顺序；val覆盖稳定。
4. 缺片、损坏、截断、错label/成员配对应明确失败。
5. 可选授权的分布式训练测试：步数一致、不挂起、save/resume和签名拒绝符合预期。
   单元测试是否执行优化器更新必须事先读源码确认，不能未经授权执行这一步。
