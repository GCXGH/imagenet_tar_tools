# Original-byte ImageNet TAR preparation

## ImageNet 加速方案导航（2026-11-10）

后续方案按目录隔离，见 [`experiments/README.md`](experiments/README.md)。
当前 TAR 的实现说明、已完成测试及 100 epoch 长跑结果见
[`experiments/tar_original/README.md`](experiments/tar_original/README.md)；
测试优先级和授权停止点见 [`experiments/ROADMAP.md`](experiments/ROADMAP.md)。
`further_explore_reference.md` 仅作为候选方案参考，不是逐项强制执行清单。

根目录的 `prepare.py` / `verify.py` / `test_prepare.py` / `prepare_imagenet.sh`
保留为已有 TAR v1 的兼容入口，不搬动、不改写；尤其 `prepare.py` 的 SHA256
已写入现有数据的转换配方，修改它会使原输出目录的转换恢复检查失败。
新方案不得导入训练仓库或另一个方案的实现；公共约定仅限实验记录协议。
下文 machine 61 的 `/mnt/data` 是历史机器配置，本机数据位置为
`/data1/xinghaitong/dataset/`，不要直接执行历史机器的准备脚本。

Independent dataset tooling: Python standard library only; no torch, no training
repository imports, no downloads, no image resize, decode or JPEG re-encoding.
Source files remain untouched. Images are copied byte-for-byte into uncompressed
POSIX USTAR archives following the public WebDataset convention: adjacent image
and integer `.cls` members with a shared unique basename.

```bash
python prepare.py --source /path/to/imagenet --output /path/to/imagenet1k_original_tar_v1 \
    --shard-samples 4096 --seed 0
python verify.py /path/to/imagenet1k_original_tar_v1
# Optional: reread original files and compare, in addition to archive verification
python verify.py /path/to/imagenet1k_original_tar_v1 --check-source
python test_prepare.py
```

On machine 61, activate your preferred Python environment and run
`bash prepare_imagenet.sh`. Destination is a sibling of the original dataset:
`/mnt/data/xinghaitong/dataset/imagenet1k_original_tar_v1`.

Classes are sorted exactly as ImageFolder class directories. Supported extensions
match torchvision ImageFolder. Source paths are scanned deterministically, then
shuffled using Python `random.Random(seed)` (val uses seed+1). All samples are
retained, including the final short shard. Training order is NOT prescribed by
this converter: it is the reader's responsibility.

Each shard has a JSONL sidecar recording source path, ID, label and SHA256 of the
original bytes, and a JSON completion record with hashes. `manifest.json` is
published atomically only once both splits complete. Rerunning the same command
resumes completed shards after checking source metadata and archive/sidecar hashes;
settings or converter-source changes require a different output directory.
Only the converter's own `.tmp` files may be overwritten. Do not edit source
data while conversion is running. Do not point at an unrelated populated output.

`verify.py` sequentially checks every image payload hash, label, unique ID and
source path, as well as full shard hashes and counts. It does not decode images;
decoder correctness and DDP sampling belong to downstream reader tests.

## 跨设备复用与 agent 交接（2026-11-09 更新）

**其他设备上的 agent 请先读 [AGENT_HANDOFF.md](AGENT_HANDOFF.md)。**
本库仍是纯标准库数据准备工具，不包含训练入口，也不会自动启动训练。

| 文档 | 内容 |
|---|---|
| [AGENT_HANDOFF.md](AGENT_HANDOFF.md) | 新设备实施顺序、权限边界、验收与交付清单 |
| [READER_INTEGRATION.md](READER_INTEGRATION.md) | TAR读取、DDP/worker分配、尾样本、shuffle/RNG和恢复要求 |
| [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) | 公平测试矩阵、完整遍历、遥测、调度与时间计算 |
| [MACHINE_61_RESULTS.md](MACHINE_61_RESULTS.md) | 本机实测：约3.4倍短测、约4.9倍整轮改善及适用限制 |
| [evidence/machine61_pipeline_summary.csv](evidence/machine61_pipeline_summary.csv) | 26项诊断的便携数值摘要 |

经验总结：机械盘/RAID上，将大量小文件读取改为原字节TAR顺序读取，并测试合适的
worker数和GPU调度，可能显著降低输入等待。但 **4 workers、4+4卡、4.9倍收益均不是跨机器默认答案**。
TAR不等于不解码，不保证提高识别精度；改变采样/RNG轨迹也不保证与ImageFolder逐位等价。

`prepare_imagenet.sh` 内置的是61机器的绝对路径，不要在其他设备直接执行。
请优先使用上面的通用CLI，并替换source/output；转换、完整校验也会产生大量磁盘I/O，
须事先确认设备负载和用户授权。新增文档不改变转换器源码或既有数据的转换签名。

## Public references and reproducibility

- WebDataset format and shard writer:
  https://github.com/webdataset/webdataset
  https://github.com/webdataset/webdataset/blob/main/src/webdataset/writer.py
- OpenCLIP epoch-seeded shard shuffle and rank/worker splitting:
  https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/data.py
- NVIDIA DALI mechanical-storage rationale for TAR and buffered sequential reads:
  https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.readers.webdataset.html

These projects were consulted on 2026-11-08. This is an independent implementation
of the public format, not a copy of their source code and not dependent on their
packages. The manifest stores our converter SHA256 and all conversion parameters;
archive metadata (mtime/uid/gid/mode) is fixed. Preserve this entire tools directory
alongside published experiments; it lives outside the training repository and is
not automatically included in that repository's git history.
