# ImageNet-1K 数据 I/O 加速实验：设备端 Agent 执行规范

> 文档用途：将本文件交给能够访问训练服务器和代码仓库的 Agent。目标是**评估多种社区认可的数据 I/O 加速方案，必须包含 FFCV**，在尽量保持 DeiT、Swin Transformer、ConvNeXt、VMamba 对照公平性的同时，找到适合约 128GB RAM、HDD、多 GPU PyTorch 训练的实用方案。
>
> **权限原则：允许自动调查、设计、实现、小样本验证；禁止未经用户明确同意下载大型数据、全量转换、长时间基准或完整训练。任何先前的概括性同意不等于本次高消耗操作授权。**

## 0. 首要目标与工作方式

1. 至少评测原生 ImageFolder、Linux Page Cache、Swin/VMamba ZIP 缓存、WebDataset TAR shards、FFCV；有条件时再测 SSD 副本、DALI，以及可选的 LMDB/WIDS。FFCV 不可被省略或以口头结论替代。
2. 先区分 **I/O-only 变化**（不改变原始 JPEG 字节和增强配方）、**采样顺序变化**、**数据内容变化**（JPEG 再编码/缩放）、**预处理算子变化**。优先保留真实、可追溯的实验配置。
3. 分开评价纯读取吞吐、含解码/增强的 DataLoader 吞吐、实际训练吞吐与训练精度。不可把孤立的 dataloader benchmark 当作模型训练加速倍数。
4. 所有方案必须可切换，尽量在数据接口层适配现有 backbone，不擅自重写模型/训练逻辑。保持现有代码仓库的 commit 和环境不变；优先新分支、独立脚本、独立 conda 环境。
5. 保留每种方案的运行条件、配置、命令、数据来源、是否通过验证，以及可复现的报告。

## 1. 明确授权边界（最高优先级）

### A. 无需额外审批，可以执行

- 只读查看 `free -h`、`lsblk`、`df -h`、`findmnt`、`nvidia-smi`、`lscpu`、`iostat`（已有时）、`du -sh`、Python/torch/CUDA 版本以及运行中作业；只读检查已有仓库和数据目录。
- 阅读官方仓库源码、配置、文档；检查当前环境包兼容性。可以检查网络文档，但不下载大型资产。
- 以极小规模现有样本做输入/输出正确性测试；默认上限为 **最多 1,000 张图像、单次最多 2 GiB 新增磁盘写入、单次最多 5 分钟运行、最多 1 GPU、最多 4 个 worker**。如有 CPU/GPU 资源竞争，进一步缩小范围。以上是本项目建议的自动执行预算，若用户另行指定，以用户限制为准。
- 可以编写转换脚本、配置、测试程序和命令草案；可以使用模拟/合成极小数据验证流程；修改只允许在单独实验目录或分支中进行。
- 可用已有、明确不影响其他任务的工具做短时测量；不能为此暂停、抢占、杀死其他任务。

### B. 必须先汇报成本并得到明确批准，然后才可执行

- 下载 ImageNet-1K、ImageNet-21K、额外预训练权重、容器镜像、GB 级安装包、数据镜像或任何未明确授权的大型数据。
- 将**完整** ImageNet 转换为 ZIP/TAR/Beton/LMDB，重新压缩/降分辨率，生成几十至上百 GB 文件，或批量复制到另一磁盘。
- 完整数据集冷缓存预热、多 epoch 遍历、长期 DataLoader 压测、超过 5 分钟或超过 1 GPU 的 benchmark、完整 backbone 训练或精度对照。
- 创建/修改大规模 RAM disk（`tmpfs`）、调用 `vmtouch -l` 等锁定大量内存的操作。
- 在正在使用的训练环境中安装或升级 FFCV、DALI、CUDA、torch 等可能影响依赖的包；新环境中依赖下载如超过预算，同样需审批。
- 删除、覆盖、搬移已有数据；清理系统缓存；执行 `drop_caches`；修改 swap、系统服务、挂载参数、I/O 调度器或内核设置；使用 sudo 修改系统状态。
- 有潜在持续花费的云端资源、商业 API 或新硬件采购。

**对审批的要求：** 在执行每个 B 类操作之前，先给出方案名称、源/目标绝对路径、将读写的数据量、预计额外磁盘占用、预计时间范围及其不确定性、CPU/RAM/GPU 占用、对共享服务器的潜在影响、可回滚办法、精确命令。等待本次操作的明确批准，不能把“请自行决定”解释成无限预算。无法估算时先完成小样本测量，给出区间而非编造精确数值。

**明确禁止：** 静默下载整个 ImageNet、静默全量预处理、清理系统 Page Cache、擅自删除或覆盖原有数据、因依赖冲突而改动生产环境、静默启动长训练。不要用 `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` 为了造出冷缓存数据，这会影响所有用户。

## 2. 先做环境与数据清单（不修改系统）

收集并报告：

- CPU 型号/物理核数、GPU 型号/数量及空闲状态、总内存/`MemAvailable`、swap、`/dev/shm` 大小、文件系统类型、各磁盘 `ROTA`、可用空间、源数据所在的真实块设备（不能仅看目录名字）。
- 原始 ImageNet train/val 的实际大小、样本数、类别数、目录结构、文件权限、是否存在 ZIP/TAR/Beton 副本；只读抽查 20–100 张图像的格式/尺寸；必要时确认官方 train 约 1,281,167、val 50,000，但应以本机数据及 manifest 为准。
- Python、PyTorch、torchvision、timm、CUDA、FFCV、WebDataset、DALI 版本及是否已安装；用 `pip show`/只读版本查询，不应直接升级。
- 当前 DeiT、Swin、ConvNeXt、VMamba、PBI 或其他实验代码仓库路径、commit、当前训练方式（DDP、batch、增强、sampler、label mapping）。
- 当前磁盘是否有其他任务读写，避免把共享负载波动误判成方案好坏。

可参考只读命令：

```bash
free -h
cat /proc/meminfo | grep -E 'MemTotal|MemAvailable|Cached|SReclaimable|Swap'
lsblk -o NAME,SIZE,ROTA,TYPE,FSTYPE,MOUNTPOINTS
df -hT
findmnt -T /path/to/imagenet
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
lscpu | grep -E '^CPU\(s\):|^Model name:|^Socket\(s\):'
du -sh /path/to/imagenet/train /path/to/imagenet/val
```

不做全量遍历/校验和直到确认本地文件规模和 I/O 影响。若数据已被其他任务占用，先评估冲突。

## 3. 必须纳入的实验矩阵

| ID   | 方案                                                        | 文件内容/采样变化                             | 重点                                                        | 执行阶段                    |
| ---- | ----------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------- | --------------------------- |
| A0   | `ImageFolder` 原始基准，2/4/8 workers（按预算）             | 原始 JPEG；原始 sampler/增强                  | HDD 传统方案性能基线                                        | 小样本先行                  |
| A1   | `ImageFolder` + Linux Page Cache 热缓存                     | 无数据变化                                    | 评估 128GB RAM 的实际受益；记录冷热状态但不执行系统缓存清理 | 小样本先行，完整预热需审批  |
| A2   | 原始 JPEG + SSD/NVMe（若已有闲置空间）                      | 不改变图片                                    | 将存储介质收益与框架收益分离                                | 全量复制需审批              |
| B1   | Swin/VMamba ZIP + `cache_mode=no/part`，`full` 仅内存足够时 | ZIP 字节无损；`part` 可能固定各 Rank 样本子集 | 官方 Backbone 的可对齐方案                                  | 小样本先行；全量 ZIP 需审批 |
| B2   | WebDataset 原始 JPEG TAR shards                             | 图像无损；shuffle/epoch 策略可能不同          | HDD 顺序读取、多 GPU 分片                                   | 小样本先行；全量分片需审批  |
| C1   | FFCV `.beton`，`write_mode='jpg'`，例如 512px/Q90           | 重新编码、可能缩小图像                        | 强制包含的高性能方案，128GB 内存的实际落地候选              | 小样本先行；全量需审批      |
| C2   | FFCV 另一组编码参数，如 1600px/Q90                          | JPEG 再编码但减少预缩放                       | 衡量压缩、细节、吞吐与公平性                                | 先估算；全量需审批          |
| C3   | FFCV `raw` 或 `smart`（可选）                               | 可能显著增大体积                              | 区分解码优化与存储优化                                      | 仅有资源和价值时再测        |
| D1   | NVIDIA DALI + ImageFolder/支持的 reader                     | 可能改变 JPEG 解码与增强实现                  | 分离 GPU/CPU 预处理加速                                     | 兼容性通过后再测            |
| D2   | LMDB / WIDS（可选）                                         | 视具体编码及 sampler 而定                     | 增补索引随机访问方案                                        | 后置，不可替代 FFCV         |

必须报告至少 A0、A1、B1、B2、C1 的实现或阻塞原因；不能因为 FFCV 安装复杂而略过，必须尝试在**隔离环境中**实现最小样本读写并准确记录阻塞点。若无用户授权，则到全量测试前停止，不可声称获得了全量性能结论。

### 对 FFCV 的特殊约束

- `RGBImageField` 的默认 `write_mode='raw'`，**不会**自动变成 JPEG Q90；显式记录 `write_mode`、`max_resolution`、`jpeg_quality`、decoder、augmentation pipeline、输出 dtype/device。
- 首选小数据测试 `write_mode='jpg'`、`max_resolution=512`、`jpeg_quality=90`，再考虑 `1600/Q90`。文献中的 ImageNet 文件体积只是参考，不能假定本机完全相同。
- 官方文档中 `OrderOption.QUASI_RANDOM` 与 `distributed=True` 不兼容。必须按**本机实际安装版本**检查，不得直接在多 GPU 测试中套用此组合；DDP 可以先采用支持的 `RANDOM`/其他合法模式，明确报告 `os_cache` 的取值及 RAM 峰值。
- 检查 `os_cache=True/False` 对缓存和 I/O 的影响；不能把 `mmap` 或 `os_cache=True` 理解为数据必然完整常驻 RAM。
- 由于 FFCV RandomResizedCrop decoder 不一定逐位重现 torchvision 的随机数、插值和增强顺序，必须做图像分布/训练配方对照。

### 对 Swin/VMamba ZIP 的特殊约束

- 从官方 `cached_image_folder.py`、`data/build.py`、配置文件追踪真实执行路径，确认 `cache_mode` 真正生效，不能只写配置不验证。
- `cache_mode=part` 常把 rank 固定映射到样本子集，与标准 DistributedSampler 每个 epoch 重新划分样本的语义不完全一致。记录此差异及是否发生尾批丢弃/重复。
- 不得误以为 8 GPU 就把总 RAM 需求除以 8。应记录所有 Rank 的总 RSS/PSS、Page Cache，以及 DataLoader worker 的对象复制开销。

### 对 WebDataset 的特殊约束

- 保存原 JPEG 字节、标签和 sample ID；转换前随机混合类别，不得直接按 class 连续打包后只用小 buffer shuffle。
- 测试不同 shard 大小（如 256MiB / 1GiB，参数需先经小样本评估）、`shardshuffle`、sample shuffle buffer、node/rank/worker split。
- 重点核对每个 epoch 样本唯一性、覆盖率、重复率、rank 间重叠、类别频次、数据尾部处理、可复现 seed。`resampled=True` 不等同于全量无重复 epoch。
- 在多 GPU 下防止分片数量不足、各 rank batch 数不一致导致 DDP hang。

## 4. 实验公平性与 Backbone 对齐

先逐一检查**本地实际使用的版本/配置**，而不是假设官方仓库默认参数就是实验配方。建立 `pipeline_manifest.json`，至少包含：

- 数据来源、ImageNet 类别映射、train/val 样本数、转换脚本 commit、编码格式、原 JPEG 是否逐字节保留。
- train/val 输入空间分辨率、RandomResizedCrop 的 scale/ratio、插值方式、RandomHorizontalFlip、color jitter、RandAugment/AutoAugment、Random Erasing、Mixup、CutMix、Normalize、验证集 resize/crop。
- `sampler` 类型、seed、drop_last、world_size、每个 rank 的样本数、epoch 样本覆盖、随机性、异常样本处理。
- batch size（per GPU / global）、梯度累积、优化器、学习率计划、AMP、EMA、epochs 等模型训练参数保持一致。

对比实验采用**两层结论**：

1. **高保真对照（研究论文默认）**：原始 JPEG 经 Page Cache、SSD、无损 ZIP/TAR 的传输，不改变原始训练增强/验证协议；可在代表性 backbone 上做基准。
2. **高吞吐对照（FFCV 等）**：允许 JPEG 重编码和增强实现变化，但原始 backbone 与新方法**必须共同使用同一改造后的管线重新训练**。不直接把 FFCV 下新方法的准确率与原始官方 recipe 得到的公开 Top-1 当作严格同条件增益。

推荐最小精度验证是选一个可运行的典型模型（如 Swin-T 或 ConvNeXt-T）：先验证同一组 JPEG 的 decode/crop 统计与类别映射，再安排用户批准的较完整训练比较；最终若发现系统性 Top-1 差异，在报告中单独列出管线差异，并尽可能用 2–3 个随机 seed 估计波动（多 seed 长训练必须逐项批准）。验证集首选保持原始 JPEG 和既定 resize/center crop 的实现，从而隔离训练数据变化。

## 5. Benchmark 统一协议

必须区分四类测量，不能混为一个数：

1. **storage-only**：读文件字节，无图像解码；测试顺序/随机存储读取。
2. **loader-only**：包括 JPEG 解码与实际训练 transforms（说明 CPU/GPU 与数据传输是否计时），不训练模型。
3. **train-step**：真实模型 forward + backward + optimizer；warmup 后计时并 `torch.cuda.synchronize()`，记录批次等待时间与 GPU 运算时间。
4. **end-to-end**：含 epoch 启停、DDP 同步、实际 checkpoint/验证环节，需审批长时间运行。

至少记录：wall-clock、images/s（所有 GPU 的全局吞吐）、batch latency p50/p95、Data Time 占比、CPU 峰值、GPU 显存/利用率、主机 MemAvailable、进程 PSS/RSS（区分共享缓存和重复进程内存）、HDD `r/s`/`rkB/s`/`r_await`/`%util`、cache warm/cold 状态、文件体积、转换时间、预热时间、失败情况。

- 每种配置尽可能重复 3 次短测，交错测试顺序，避免缓存和其他服务器任务引起偏差；只有用户批准才能扩大测试规模。
- 不得为了公平而清理全系统 Page Cache；只记录可观察的实际缓存状态。不能保证 cold cache 时标注 `cold_uncontrolled`，而不是冒称严格冷缓存。
- 报告吞吐结果必须附带硬件、world size、workers、batch size、压缩格式/分辨率/增强、采样模式；只在**相同测试定义**下比较加速倍数。
- 指出瓶颈：HDD seek、解码、增强、CPU、内存、H2D、GPU 或 DDP；避免机械地把所有耗时归因于 I/O。

建议汇总表：

| ID   | 方案         | 全局 images/s | 相对 A0 | RAM 峰值 | HDD 读取 | 体积 | 数据/采样变化      | 状态   |
| ---- | ------------ | ------------: | ------: | -------: | -------: | ---: | ------------------ | ------ |
| A0   | ImageFolder  |           TBD |   1.00x |      TBD |      TBD | 原始 | 无                 | 未运行 |
| A1   | Page Cache   |           TBD |     TBD |      TBD |      TBD | 原始 | 无                 | 未运行 |
| B1   | ZIP + part   |           TBD |     TBD |      TBD |      TBD |  TBD | rank 子集固定      | 未运行 |
| B2   | WebDataset   |           TBD |     TBD |      TBD |      TBD |  TBD | shuffle/epoch 变化 | 未运行 |
| C1   | FFCV 512/Q90 |           TBD |     TBD |      TBD |      TBD |  TBD | JPEG 重编码+缩放   | 未运行 |

## 6. 执行里程碑和每阶段停止点

**Phase 0：只读调查。** 输出机器/数据/代码清单、候选方案兼容性、预计容量和初步风险。没有原始数据时，不得自行下载；请求用户指出合法可用的已有数据位置。

**Phase 1：准备隔离框架。** 在独立目录做可切换 data backend（`imagefolder`/`zip`/`webdataset`/`ffcv`/`dali`），共享一套 manifest、样本 ID、标签校验和指标脚本。先以模拟或少量已存在图像验证。

**Phase 2：轻量 smoke tests。** 每个 backend 至多 1,000 张图像、5 分钟、1 GPU、2 GiB 新写入。必须确认：无标签错位、无无意重复、图片可正常解码、DDP 模式可被设计/模拟、程序可退出。隔离 FFCV 安装若会下载大量包则先审批。

**Phase 3：提交批准请求，不启动高负载。** 分别列出全量 ZIP、WebDataset、FFCV 512/Q90、FFCV 1600/Q90 的预计输入量、输出量、时间、CPU/RAM/GPU 使用、位置和清理方案；推荐一次只批准一个转换实验，转换完成前不要无限接续下一项。

**Phase 4：在用户逐项批准后执行全量转换和中等规模 benchmark。** 每次转换校验 manifest、样本数、标签映射、错误样本日志和文件体积；生成对照表。转换失败应保留诊断，不删除原始数据。

**Phase 5：精度对齐。** 先做低成本图像/增强一致性诊断；准备正式 backbone 与方法对照训练方案。**再次请求批准**训练时长、GPU 数量、epoch、总 GPU-hours、输出与 checkpoint 预算，批准后才运行。

## 7. Agent 应产出的文件

在独立实验目录产出：

- `environment_report.md`：硬件/磁盘/数据/当前训练参数/资源竞争情况。
- `compatibility_matrix.md`：各库版本、依赖与安装风险，FFCV 的可运行/不可运行原因。
- `pipeline_manifest.json`：每条管线的图像/label/采样/增强参数和数据哈希抽查。
- `benchmark.py` 与若干 backend adapters：可配置、小规模优先、默认 dry-run，不允许默认下载或全量转换。
- `results.csv` 和 `results.md`：实测原始数据、图表、异常、比较与可信度说明。
- `approval_requests.md`：所有尚未执行的高耗时/高占用操作的确切命令与预算。
- `recommendation.md`：最终给出（a）论文高保真方案，（b）128GB 机器上的最优性价比方案，（c）极致吞吐 FFCV 方案，分别说明代价和精度风险。

## 8. 文献/官方入口（实现前再核验版本）

1. Swin ZIP/缓存官方说明：https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md
2. Swin 参数和 cache_mode：https://github.com/microsoft/Swin-Transformer/blob/main/main.py
3. Swin 数据加载源码：https://github.com/microsoft/Swin-Transformer/blob/main/data/build.py
4. Swin 缓存源码：https://github.com/microsoft/Swin-Transformer/blob/main/data/cached_image_folder.py
5. VMamba ImageNet 加载说明：https://github.com/MzeroMiko/VMamba/blob/main/assets/get_started.md
6. DeiT 官方仓库：https://github.com/facebookresearch/deit
7. ConvNeXt 官方训练说明：https://github.com/facebookresearch/ConvNeXt/blob/main/TRAINING.md
8. FFCV 总文档：https://docs.ffcv.io/
9. FFCV 编码格式：https://docs.ffcv.io/working_with_images.html
10. FFCV 参数及 DDP 约束：https://docs.ffcv.io/parameter_tuning.html
11. FFCV ImageNet 体积/吞吐：https://docs.ffcv.io/benchmarks.html
12. WebDataset 官方仓库：https://github.com/webdataset/webdataset
13. WebDataset FAQ：https://github.com/webdataset/webdataset/blob/main/FAQ.md
14. NVIDIA DALI PyTorch：https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/ndd_plugins/pytorch_usage.html
15. PyTorch DataLoader 官方：https://docs.pytorch.org/docs/stable/data.html

## 9. 初次回复用户的格式（Agent 必须遵守）

先完成 Phase 0 的只读调查。第一次回复必须包含：

1. 机器可用 RAM、源数据是否真在 HDD、磁盘空闲空间、GPU 当前忙闲和 Python/torch 环境。
2. 本次能立即安全执行的最小实验清单，以及所有因审批而暂缓的工作。
3. 各格式（ZIP、TAR、FFCV）全量转换所需空间/时间的估算方法与尚未确认的参数，不要凭公开数据捏造本机时间。
4. 下一项最值得批准的操作，附上成本上限、回滚方案和完整命令。

完成后停止在审批边界，不要自行进入全量转换或长时间训练。