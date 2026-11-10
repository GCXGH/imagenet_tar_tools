# WebDataset 标准化基线：阶段1

阶段2已完成256张现有图片的限额CPU处理/IPC检查，以及单卡ViT-B/16正反序训练短测。
结论和限制见 [PIPELINE_FINDINGS.md](PIPELINE_FINDINGS.md)：worker批量输出显著优于
本原型逐样本IPC，但不是当前生产训练可再获得的收益；uint8降低传输成本却未提高总吞吐。
本次输入压缩字节留在RAM，不能称作完整TAR流式训练或全量HDD性能结果。

目标是可迁移的数据层，而非针对某个模型/服务器优化。复用原字节TAR，
训练代码仅传入root、transform、batch、rank/world、epoch，不导入任何模型或其他方案。
**当前是正确性优先的实验原型，不是经性能验证的生产替代。**

## 已完成

- WebDataset 1.0.2 + braceexpand 0.1.7隔离安装到
  `/tmp/opencode/imagenet_webdataset_deps_v1`，wheel合计80,879 bytes；
  numpy/PyYAML/Pillow/torch复用调用环境，没有修改spb安装内容。
  临时依赖目录不保证持久保存；复现时用本目录requirements安装到自己的隔离环境。
- 使用WebDataset DataPipeline、tar_file_expander、group_by_keys和shuffle。
  自己管理本地文件的with生命周期：所测版本的WebDataset便捷开流路径在测试中
  出现ResourceWarning，组合公开stages后验证无该警告。没有修改第三方包。
- 接受现有manifest，保留原图片字节、label、sample key，不resize/reencode。
- rank分配整片、worker读取不重叠的shard子集。先确定性epoch打乱，再按sample计数贪心分配。
- 支持两种显式有限epoch，不repeat、不resample、不静默跳坏图：
  - `coverage`：所有样本恰好一次，允许各rank batch数不同和最后partial batch；适合评估/读取。
  - `equal_steps`：按最小rank的完整batch数截断，各rank等步数；用于普通同步DDP训练。
    丢弃量在`loader.plan`报告，不能声称全覆盖。
- 在主进程组batch，避免IterableDataset按worker丢尾部；代价是逐样本IPC，尚未调优。
  pinning关闭、worker非persistent，每个epoch重建loader以显式传播epoch/seed。

## 用法

从本方案目录导入，不需要模型/训练仓库：

```python
from loader import EpochLoader

# transform由具体任务提供；通常输出相同尺寸的Tensor。
loader = EpochLoader(
    root='/path/to/imagenet_tar', split='train', transform=train_transform,
    batch_size=256, world=world_size, rank=rank, workers=4,
    seed=0, epoch=epoch, shuffle_buffer=512, policy='equal_steps',
)
print(loader.plan)  # 明确每rank可用/实际发出/全局丢弃量
for images, labels, keys in loader:
    # 保留调用方的Mixup/CutMix、device transfer和训练逻辑
    pass
```

评估用`coverage`，但不能直接在不等步数rank上执行带逐步collective的DDP forward；
应使用无逐步同步的评估模型，循环结束后再统一reduce。训练选equal_steps。
`transform=None`时返回PIL图像，调用方需提供合适collate_fn，不可用默认Tensor collation。
同worker布局+seed+epoch可复现；不承诺改变worker数后随机增强或输出顺序不变。
当前只支持本地shard，未实现URL/对象存储、任意新manifest格式、自动恢复或中途batch恢复。

## 验证记录（2026-11-10）

```bash
PYTHONPATH=/tmp/opencode/imagenet_webdataset_deps_v1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 /data1/xinghaitong/usr/anaconda3/envs/spb/bin/python \
  -W error::ResourceWarning experiments/webdataset_baseline/test_loader.py

# 默认dry-run；execute才读取，60秒时限，最多512张，无GPU。
PYTHONPATH=/tmp/opencode/imagenet_webdataset_deps_v1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 /data1/xinghaitong/usr/anaconda3/envs/spb/bin/python \
  experiments/webdataset_baseline/smoke.py \
  --root /data1/xinghaitong/dataset/imagenet1k_original_tar_v1 --execute \
  --result experiments/webdataset_baseline/smoke_20261110.json
```

7项合成测试通过（约0.77秒）：字节/标签、0/2 workers下模拟两rank的全覆盖/不重叠、
等步数/丢弃计数、epoch可复现、空rank/worker、错误计数与标签拒绝、数据不足拒绝。
这里的rank是顺序模拟，没有启动Gloo/NCCL进程组，不代表真实DDP集成测试已通过。

128张真实ImageNet检查通过（约3.10秒）：14,142,589 bytes有效payload，
与源文件字节和PIL RGB像素相同，标签/key与JSONL一致。原始文件/TAR均只读，未构造真实图片子集。
时间包含源文件重读、hash和两次decode，仅是功能检查，不是吞吐结果。
未验证随机训练增强分布或训练精度，也没有进行全量遍历/性能基准。

只读取manifest的四卡batch256 epoch0计划对照：

| 策略 | rank可用样本 | 每rank batch数 | 全局丢弃 |
|---|---|---|---:|
| coverage | 319488 / 322703 / 319488 / 319488 | 1248 / 1261 / 1248 / 1248 | 0 |
| equal_steps | 同上，实际各发319488 | 1248 / 1248 / 1248 / 1248 | 3215（约0.251%） |
| 现有专用reader | 划分sample range而非整片 | 1251 / 1251 / 1251 / 1251 | 143 |

说明成熟库不自动解决epoch语义。不能保持原1251 updates假设不变而直接换loader；
学习率schedule/epoch样本数应显式记录。整片分配减少访问耦合，但有负载不均取舍。

## 后续关卡

1. 示例增强的限额处理/IPC对照已做；调用方实际配方与完整TAR流式训练对照仍未做。
   本正确性原型逐样本传输，不能代表WebDataset的最优性能。
2. 对代表性模型做小规模真实DDP接口验证，确认训练步数、评估及epoch重建；
   长时间/多卡必须提交准确命令和资源预算后等批准。
3. 若需要严格全覆盖+等步数，应单独设计跨片batch/sample范围或其他协议，
   不默认repeat/resample来掩盖计数差异。

随后再进入DALI（解码/增强）或FFCV（一体化）阶段，不自动安装/转换其他方案。
需要新建或下载大型数据集时立即停止申请，不利用本次小测授权扩大范围。
