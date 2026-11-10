# 本机只读快照

采集：2026-11-10 约 15:57–16:02 CST。资源随其他作业变化，执行前必须重新检查。

- CPU：2 × Xeon Platinum 8180，56 物理核 / 112 逻辑 CPU，2 NUMA nodes。
- RAM：约 93 GiB；MemAvailable 约 74 GiB；swap 975 MiB 已占满。
  不把 Cached 与 MemAvailable 相加；不将全部可回收缓存视作独占实验预算。
- `/data1`：`/dev/sdb1`，ext4，约 6.5 TiB，剩余约 765 GiB，88% 使用。
  块设备上报 ROTA=1；底层是否硬 RAID、盘数及条带结构未确认，不能据此虚构物理盘性能。
- 根盘 `/dev/sda2`：剩余约 15 GiB，99% 使用，ROTA=1。没有发现可用 NVMe/SSD 挂载。
  `/tmp/opencode` 位于根盘，不适合新建大副本、庞大依赖缓存或容器。
- `/dev/shm`：约 47 GiB 容量；与主机 RAM 共用，不是额外内存，不建立全量 RAM disk。
- GPU：8 × RTX 4090。GPU0 占用约 20.9 GiB、GPU1 约 18.2 GiB 且有计算、
  GPU2 约 8.8 GiB；GPU4–7 在此次查询时空闲。不以显存空闲作为充分授权。
- 拓扑：GPU0–3 同组 PIX/NUMA0；GPU4–7 同组 PIX/NUMA1；组间 SYS。
  GPU4–7 的 CPU affinity 为 `28-55,84-111`。
- 一次 1 秒 iostat 区间中 sdb 约 11.4 MiB/s、273 read IOPS、r_await 3.95ms、
  util 92.4%。这是其他作业存在时的短时快照，不是该存储的带宽上限，也不是 TAR 压测结果。

## 数据与环境

数据目录仅做顶层检查和读取既有 manifest，未重新扫描所有原始小文件：

- `/data1/xinghaitong/dataset/unpacked_imagenet/`：原始 ImageFolder，保留。
- `/data1/xinghaitong/dataset/imagenet1k_original_tar_v1/`：完整、已校验 TAR。
- `/data1/xinghaitong/dataset/imagenet1k_original_tar_pilot_v1/`：诊断子集，不能代表冷缓存全量。
- `/data1/xinghaitong/dataset/imagenet_tar_preparation_logs/`：既有转换日志。

现有 `spb` 环境（仅查询 distribution metadata，未安装或升级）：

| 包 | 版本 |
|---|---|
| Python | 3.12.15 |
| torch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| timm | 0.4.12 |
| Pillow | 13.0.0 |
| FFCV / WebDataset / LMDB | 未安装 |
| DALI cuda110 / cuda120 distribution | 未安装 |

现有 TAR converter 是标准库实现，reader 用 PyTorch/PIL，不依赖 WebDataset 包。
FFCV/DALI 与本机 Python/CUDA 的具体兼容组合尚未通过安装实测；不得宣称已兼容或一定不兼容。
不修改生产环境；如需新环境，先报告下载量、位置和空间预算。

PBI wider 100e 已于 2026-11-10 15:51 完成；其他作业仍存在，不能因此假定整机空闲。
