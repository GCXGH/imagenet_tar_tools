# DALI（已安装，阶段1功能小测通过）

## 最新：解码路径消融已完成

见 [DECODE_ABLATION.md](DECODE_ABLATION.md)。同reader/GPU增强/训练步，
CPU decode一轮442.34秒（post-first .33049秒/步），mixed464.77秒（.34798），
最后运行原TAR619.10秒（.46038）。CPU比mixed吞吐高约5%、少1.08GiB/卡显存。
CPU DALI比反向原TAR整轮吞吐高约40%；GPU JPEG解码不是获得收益的必要条件。
尚不能将剩余收益单独归因GPU增强，需要进一步拆解处理/传输。所有任务结束。

## 扩大测试（已完成）

三项完整epoch均成功结束，结果见 [EPOCH_RESULTS.md](EPOCH_RESULTS.md)。
原TAR624.94秒、PIL对照623.42秒、DALI458.99秒；DALI整轮吞吐比原TAR高36.16%，
首步后高33.35%，GPU平均util约67.2%→82.6%，总显存最大采样约19.95→21.69GiB。
这是整套pipeline的单次性能结果，不是精度等价/纯I/O归因；采样顺序、算子及缓存不同。
所有测试已退出，不再自动启动任务。

用户明确要求四卡全局batch1024完整epoch对照。详见 [EPOCH_TEST.md](EPOCH_TEST.md)。
原生reader首批256图payload hash/label检查通过，索引合成单测通过，
四卡PBI wider 12步预检梯度有限。随后启动original / pil_matched / dali三项串行一轮，
每项1251更新，30分钟时限，不保存checkpoint，不修改正式训练代码。
结果根：`/data1/xinghaitong/results/imagenet_io_benchmarks/dali_epoch_20261110/`；
调度日志：`/tmp/opencode/dali_epoch_suite_20261110.log`。
执行前计划和限制保留于EPOCH_TEST.md；当前完成状态以上方结果报告为准。
有效索引约82MiB，初版诊断索引约47MiB保留，不重写原始TAR。

## 当前状态

2026-11-10在用户明确授权后，已在`spb`安装DALI cuda120 2.4.0。
确认原环境没有DALI：`nvidia.dali`不可发现，pip/conda无DALI包；
原有nvidia CUDA distributions是torch依赖，不是DALI，PyTorch通常不自带DALI。

- 精确依赖见`requirements-tested.txt`，安装报告含wheel URL/hash/元数据：
  `install_records/installed_20261110.json`。
- 只新增13包，torch2.7.1+cu118、torchvision0.22.1+cu118、numpy2.5.4、packaging26.3未变。
  `pip check`通过。没有升级driver或修改系统CUDA/LD_LIBRARY_PATH。
- wheel合计498.90MiB，SHA256全部校验；首轮240秒下载超时未安装，
  用户随后批准15分钟续传/安装，实际约411秒完成。
- 下载文件保留在`/data1/xinghaitong/.cache/dali_install_tmp/verified_wheels`，
  不在根盘缓存、不删除用户文件；未自动清理这些新增文件。
- DALI CUDA12与torch cu118在同进程GPU运行成功；仅证明所测路径共存，
  不代表所有算子和复杂训练配置均兼容。

## 阶段1：128张真实图像、单GPU4、无新数据副本

独立`smoke.py`直接读取已有TAR的少量prefix并校验hash/label，
压缩字节在RAM（15,758,205 bytes），无需WebDataset、生产仓库或其他方案依赖。
最多256图/4 CPU threads/单卡，240秒报警，默认dry-run，result不覆盖。
当前外部输入只验证处理后端，还没有测试DALI readers.webdataset直接流式读取。

| 检查 | 结果 |
|---|---|
| CPU decode vs PIL RGB | 128/128像素完全一致 |
| mixed decode vs PIL RGB | 尺寸/标签一致；128张均有像素差异 |
| mixed像素差 | 每图mean abs均值0.550（0..255单位），最大44 |
| GPU resize/crop/normalize | 输出`[16,3,224,224]`、有限；与PIL非逐位一致 |
| 确定性预处理差异 | Normalize后mean abs0.1092、max abs3.869；尚未拆解来源 |
| DLPack到torch GPU | 成功，保留DALI输出所有者直到训练步结束 |
| ViT-B/16完整训练 | weights=None、eager BF16、SGD、batch16、8步，loss有限 |

预处理比较包含mixed decode、bicubic resize、尺寸/取整、center crop等差异。
不能因为mean decoder误差较小就推断整管线相等，也不能未经拆解称为“正常可忽略”。
下一步需分离CPU/mixed decoder、相同resize输出尺寸及crop坐标/取整和插值实现，
再设计随机训练增强对照。

短训练首步0.352秒，后7步约0.066–0.082秒，约217张/秒。
这是**功能小测，不是加速结果**：确定性resize256/center224，没有RRC/flip/RandAugment，
每步同步，短时间/无多次交错重测；与前一个WDS示例随机增强基准不可直接算加速比。
decode记录的0.82/0.84秒包含回CPU和PIL比较，也不是纯decode测速。
没有采集GPU utilization序列，不报告利用率改善。没有精度/收敛、DDP或全流式吞吐结论。

```bash
conda activate spb
# 不加execute时仅显示预算
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 python experiments/dali/smoke.py \
  --root /data1/xinghaitong/dataset/imagenet1k_original_tar_v1
# 实际执行加--execute；--result使用不存在的新JSON路径
```

## 下一步与回滚边界

先核对确定性预处理差异，再在同一输入集合、同类增强和相同计时口径下测CPU/mixed与PyTorch。
之后才测试已有TAR的DALI原生reader。不得为了测试自动生成全量索引或新数据集。
多卡/长测/全量转换仍需单独确认。本次所有安装和小测进程已退出。

如用户希望撤销安装，安装报告列出的13个新增distribution为精确卸载候选；
必须重新检查是否被新任务依赖后再卸载。不删除或更换原有torch/CUDA包，
本次不执行任何卸载/清理操作。

## 历史：只读兼容性调查（安装前，2026-11-10）

PyPI元数据：cuda110最新1.50.0；cuda120/cuda130最新2.4.0。
cuda120 2.4.0的x86_64 wheel约404.81MiB，另有nvimgcodec/nvcomp等依赖；
完整下载/解压预算尚未解析，不能只把wheel体积当作全部安装成本。
官方当前安装文档说明CUDA12构建的driver兼容下限为525.60，
本机driver535.216.01满足该文档下限，但仍需检查glibc、实际CUDA库和与现有
torch cu118的互操作，**不代表本机安装一定可用**。CUDA13构建要求更高driver，不选。
下一步应先确定具体版本与隔离环境/依赖预算，避免为测试升级现有torch/driver。

当时只查询PyPI JSON和官方文档，未下载DALI wheel、安装环境或构造数据。
考虑依赖体积与根盘空间，停在大依赖安装前，给出完整预算后再申请。
官方文档：https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html

目标：不重新编码现有图片，检查CPU/mixed JPEG解码与GPU预处理能否降低CPU路径成本。
优先调查现有TAR能否直接由目标版本readers.webdataset读取；reader的sharding、
shuffle、last_batch/padding、索引要求必须逐项验证，不能假定与当前epoch策略等价。
若需全量索引扫描，也先申请。独立环境，不更换spb中的torch/CUDA。

首个小测只替换decoder，记录CPU/mixed、nvJPEG fallback、图片格式覆盖、GPU显存；
随后才比较增强。不要把整套RandAugment替换成简单crop然后称同配方加速。
PBI基线峰值约18.8GiB，4090剩余空间有限，mixed decode缓存/队列必须实测，
GPU预处理也可能与模型竞争SM/带宽。loader-only快不保证训练快。

最多1,000图/单卡/5分钟的检查先做预算；多卡、长测、全量转换以及大依赖下载均需批准。
以上为安装前计划；当前安装版本和已验证范围以上方阶段1记录为准。

待核验官方入口：
https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/operations/nvidia.dali.fn.readers.webdataset.html
