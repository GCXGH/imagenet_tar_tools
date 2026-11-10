# 阶段2：处理管线与GPU供给的限额诊断

日期：2026-11-10。目标是验证通用数据管线设计，不调当前训练仓库。

## 结论与适用范围

1. 数据管线应同时考虑组batch、IPC、解码/增强、H2D，不应只优化磁盘格式。
2. 正确性原型的逐样本IPC有明显成本，应使用worker批量输出或等效批量传输。
   **当前生产训练本来就在worker组batch，因此这不是生产训练可再获得的2倍加速。**
3. uint8确实降低传输阶段成本，但不保证总训练更快；不能根据loader-only决定默认配置。
4. WebDataset标准化工作的主要价值仍是可迁移接口和明确采样语义，
   这里没有证明WebDataset库比原reader快，也没有测完整TAR流式加载+训练。

## 实验定义

- 现有TAR前4片各选64张，共256个原始样本，压缩字节29,851,472 bytes（约28.47MiB）。
- 验证标签、key、hash及源文件字节后，将压缩字节保存在RAM。
  后续比较故意排除磁盘，隔离CPU处理/IPC/传输；不是新建数据集，也不是冷存储基准。
- 独立`benchmark_small.py`，CPU最多4 workers；GPU只用物理4一张RTX4090。
  每条命令默认180秒硬时限（最多240秒），默认dry-run，无后台任务。
- torchvision的示例ImageNet增强：RRC224 bicubic、flip、RandAugment(2,9)、
  PILToTensor、float/Normalize；**不是现有timm配方的精确复现**，不含Erasing/Mixup/CutMix。
- 单卡模型：torchvision ViT-B/16，weights=None，无权重下载；eager BF16、SGD，
  batch16，每方案4步预热+8步计时。模型初始参数每次重置。
- 正序/反序各一次，以观察顺序影响；时间仍短，不足以代表持续训练或置信区间。
  GPU测试计时是完整forward/backward/optimizer，CUDA末端同步。

## CPU短测（含worker启动，输入压缩字节在RAM）

| 路径 | 吞吐 images/s |
|---|---:|
| PIL解码，单线程 | 488 |
| torchvision CPU解码，单线程 | 550 |
| PIL+增强+float，0 workers | 172 |
| PIL+增强+float，4 workers，逐样本IPC/主进程组batch | 231 |
| PIL+增强+float，4 workers，worker组batch | 458 |
| PIL+增强+uint8，4 workers，worker组batch | 575 |

仅前32张decoder像素对照max/mean差为0；不能推广到所有格式/CMYK图片。
torchvision decoder只测了解码阶段，尚未把后续增强迁移到tensor，因此没有整管线12.6%加速结论。
uint8的CPU计时不含Normalize，float包含；它是工作迁移而非免费省略，训练测量将其补到GPU。
逐样本seed让比较布局使用相同增强，不是建议生产环境逐图重新manual_seed。

## 单卡完整训练步短测

| 输入路径 | 正序 images/s | 反序 images/s | 秒/步范围 |
|---|---:|---:|---|
| GPU合成输入 | 362.7 | 374.5 | .0427–.0441 |
| RAM原图+CPU增强，逐样本IPC | 138.0 | 130.9 | .1159–.1222 |
| RAM原图+CPU增强，worker组float batch | 324.9 | 324.2 | .0492–.0493 |
| RAM原图+CPU增强，worker组uint8 batch，GPU normalize | 318.3 | 314.8 | .0503–.0508 |

同图像集合和标签；主进程逐样本与worker组batch的样本顺序可不同，不是同顺序优化器轨迹。
合成输入的图像和标签不同，只是计算参照，不是严格重放同一训练步。
各方案loss有限；没有精度/收敛结论，没有DDP、compile或正式实验配置外推。

组batch后达到合成参照约87–90%的吞吐，说明该小实验可以较充分供给GPU；
不能将差值全部算磁盘，磁盘在这里已被排除。正反序结果方向一致。
uint8的H2D+预处理事件约2.3ms vs float约6.3ms，然而模型阶段事件更长、
总步时略差，具体原因尚未剖析（事件区间也包含CPU提交空隙）。不据此认定GPU normalize导致模型变慢。
不使用逐事件时间相加来分摊所有重叠成本，不把CPU wait直接理解成GPU空闲时间。

## 对GPU utilization观察的解释

本次没有采集短测的nvidia-smi utilization序列，不虚构GPU占用变化。
nvidia-smi GPU utilization主要反映采样窗口内是否有kernel执行，
不是SM占用、Tensor Core效率或已达到峰值算力的比例。
低值可提示间隙，但原因可能是供给、H2D、Python/launch、同步和模型粒度。
后续评价以真实吞吐/step time、时间线空隙、CPU/RAM/GPU成本为主，而不是追100%。

## 下一阶段

WebDataset正确性原型不直接进生产；若要高效生产接口，应保留worker批量输出，
同时解决全覆盖/等步数/尾部协议，而不是单纯开启DataLoader drop_last。

通用探索下一重点是DALI解码/增强加速（复用已有原字节TAR优先），
然后FFCV一体化方案。先做依赖/算子/内存可行性调查，安装大依赖或构造大数据前停止申请。
真实多卡/长测同样先提交预算。当前没有启动DALI/FFCV安装或全量转换。

## 复现

代码：`benchmark_small.py`；三份结果：`cpu_probe_20261110.json`、
`train_probe_20261110.json`、`train_probe_reverse_20261110.json`。
环境沿用阶段1的隔离WDS依赖；torch2.7.1+cu118、torchvision0.22.1+cu118。

```bash
# 从imagenet_tar_tools根目录，默认不加execute可先看预算。
PYTHONPATH=/tmp/opencode/imagenet_webdataset_deps_v1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 /data1/xinghaitong/usr/anaconda3/envs/spb/bin/python \
  experiments/webdataset_baseline/benchmark_small.py \
  --root /data1/xinghaitong/dataset/imagenet1k_original_tar_v1 --phase cpu

# GPU阶段同一命令改phase=train，CUDA_VISIBLE_DEVICES设为明确空闲的一张卡；
# execute才执行，result需新路径不覆盖，--reverse反转GPU案例顺序。
```

新增合成单测`test_benchmark_small.py`通过：不同组batch方式按key对应Tensor相同、
不重不漏、uint8转换Normalize与float结果一致（CPU容差1e-6）；未验证GPU逐位一致。
原7项loader测试通过，生产文件未修改。
