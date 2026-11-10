# 测量、隔离与授权约定

## 安全边界

- 不下载数据集或权重来测速；直接使用本机合法已有数据。
- 用户确认前，不开启长期后台处理、全量转换/复制/预热、多卡 benchmark 或训练。
- 后续轻量正确性检查的建议上限：最多 1,000 张现有图片、2 GiB 新写入、
  5 分钟、1 GPU、4 workers；这是自设上限，不是消耗这些资源的必需量。
  当前本轮不做真实图片性能测试；实际存在资源竞争时继续缩小或暂缓。
- 不安装/升级 `spb` 环境，不执行 sudo、drop_caches、内存锁定、系统调参或删除旧数据。
- 任何超出预算、长时间处理或大下载：先列方案、精确命令、源/目标绝对路径、
  读写量、峰值磁盘/RAM/CPU/GPU、时限、潜在干扰、失败与回滚办法，等待明确批准。
- 批准一次任务不代表转换完可自动接续长测/训练。每项要有超时和明确停止点。

## 四层测量

1. **encoded-read**：读取原字节，不解码、不增强；单独标记是否包含 hash 校验。
2. **decode / transform / loader**：分别测 PIL RGB、实际训练增强、batch 组装及 IPC。
3. **H2D / train-step**：标明输出 dtype、pinning、传输/同步计时边界、模型和 DDP；
   吞吐以最慢 rank 的完成时间为准，不把 rank0 等待当作全部系统等待。
4. **end-to-end**：epoch 启停、验证、保存及编译开销。长测/正式训练单独审批。

CUDA events/同步本身可能扰动重叠：分段诊断和未插桩吞吐分别报告。
现有真实步时减合成步时不是精确 I/O 耗时；loader-only 提升也不是训练加速倍数。

## 公平性和正确性

- 相同输入 manifest 记录 split、sample ID、source path、label、payload hash。
  对照分别记录选中 ID 集合哈希和输出顺序哈希；只有集合/顺序确实相同才称同样本/同顺序。
- 原字节、decoder、增强、输出精度和 sampler 变化分别列明。
  torchvision/timm 的增强参数、插值、验证 resize/crop、Mixup/CutMix 位置必须可追溯。
- 检查 seed/epoch、rank/worker 分配、尾部、不重复、不补齐、更新次数及验证完整覆盖。
- 对 persistent workers 特别测试 epoch/seed 传播；主进程 set_epoch 不会自动更新已存活 worker 副本。
- 不用断言图像像素完全一致来替代随机增强分布检查；修改配方后的精度结论需共同重新训练。

## 记录字段

方案/代码 SHA、数据 manifest SHA、环境、CPU/NUMA affinity、GPU物理编号、
world size、batch/accumulation、worker/thread/prefetch/shuffle、存储位置、
输入量/输出量、编译与预热、计时边界、images/s、latency p50/p95、各 rank 等待/step、
RSS/PSS/MemAvailable、共享缓存、GPU 显存/利用率、iostat、样本覆盖/异常、转换时间。

不驱逐全局缓存。缓存无法受控时标 `cold_uncontrolled` 或 `cache_uncontrolled`；
tiny pilot 易热缓存，只能证明功能和缓存条件下的吞吐，不能代表全量 HDD。
允许时交错顺序并重复短测，记录其他作业；不把 ROTA/util 当作硬盘带宽的直接证据。

## 容量与成本估算

原 TAR archive 合计 156,001,105,920 bytes（约 145.29 GiB，不含 sidecars）。
ZIP_STORED 全量输出大致为原图片 payload 加目录开销，不能默认压缩 JPEG 有显著收益。
FFCV 的再编码尺寸、raw/smart 比例和分布必须由代表性小样本测得。
按类别/原文件大小/分辨率分层抽样，报告输出输入比的分位数和均值，
用全量文件数外推区间并加索引/临时文件/安全余量。小样本时间可能受热缓存高估，
还要计入同盘读写争用、校验的额外遍历和恢复扫描成本；未测前不承诺全量小时数。
