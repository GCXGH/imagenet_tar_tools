# 给其他设备 agent 的交接：原字节 ImageNet TAR

更新：2026-11-09。目标是复用已验证的方法，不照搬61机器的最优参数。

## 1. 范围与安全边界

- 本库只负责 ImageFolder → 原字节、无压缩 TAR 的转换和校验；Python标准库即可，
  不依赖torch，不包含完整训练器或GPU诊断运行器。新增材料是集成协议和实测摘要。
- 不修改/删除源ImageFolder，不重编码JPEG、不预先resize，不覆盖旧实验、checkpoint或日志。
- **准备代码不等于授权启动训练。** 没有明确授权，不启动GPU诊断、优化器更新、正式训练，
  不停止/重启已有任务；不因为转换完成而自动排队启动后续任务。
- 不全局清系统缓存、不关闭swap、不安装包或修改系统设置来制造加速结果。
- 转换和全量verify都可能重占磁盘：即使无GPU也应确认授权、空间和运行窗口。
- 现有转换参数或`prepare.py`源码变化会影响恢复校验；不要为迁就旧输出改签名。
  新输出使用独立目录。此次更新仅增加文档和证据摘要，没有修改转换器。

## 2. 新设备执行顺序

### A. 只读盘点

记录：数据所在磁盘/挂载点、HDD/SSD/网络存储、RAID逻辑盘、CPU/NUMA、RAM、
GPU型号/数量/互连、现有负载、Python/torch/torchvision/timm/PIL版本、目标有效batch。
不要将RAID逻辑盘的busy=100%直接解释为所有物理盘饱和。
原始val应已按类别目录整理，与train的类别映射一致；不能把未整理的官方val平铺目录直接当输入。

### B. 准备数据（用户确认后）

在库目录，用本机现有环境；以下路径是占位符，必须替换：

```bash
python prepare.py --source /data/imagenet --output /data/imagenet1k_original_tar_v1 \
  --shard-samples 4096 --seed 0
python verify.py /data/imagenet1k_original_tar_v1
# 可选：重读源文件，额外核验源字节，需要更多I/O
python verify.py /data/imagenet1k_original_tar_v1 --check-source
```

不要直接执行硬编码61机器路径的`prepare_imagenet.sh`。4096样本/片是已用配置，
不是所有盘和分布式布局的最优值。保留原数据、manifest、class_to_idx、所有sidecar和完成记录。
转换具备受校验的断点恢复；只有完整`manifest.json`发布后才供正式reader使用。

### C. 集成读取器，不改训练配方

按[READER_INTEGRATION.md](READER_INTEGRATION.md)对接目标仓库。
保留原后端默认行为，TAR显式启用，新run目录，不静默改batch、精度、LR、增强或模型。
本次61机器后来采用QK norm是单独的稳定性实验决策，**不是TAR输入优化的组成部分**，
不要把QK norm同步加到其他训练项目以“复用提速”。

### D. 正确性先行

- 在微型数据上测试转换/完整校验、字节/RGB/固定seed增强一致性。
- 测试多rank、多worker、不可整除的尾部、每rank相同train步数、全量val无重无漏。
- 测试epoch变化、固定布局复现、损坏文件报错、epoch边界恢复和跨签名恢复拒绝。
- 核验目标训练仓库的测试是否会调用backward、optimizer.step、真实训练或GPU。
  **不要凭“test”文件名就把全套训练测试当无训练检查执行。**
- 本库`python test_prepare.py`仅在`/tmp/opencode`建微型假图片字节，验证转换，不调用训练；
  该临时父目录须预先存在且可写，测试不会验证图片可解码性。

### E. 有界性能测试（另行授权）

按[BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md)测相同worker的两种后端，
再扫workers1/2/4，复测优胜方案，最终至少完整遍历一次训练与验证。
有多个模型/任务时比较并行与顺序的总完工时间，不只看单模型最快step。
61机器实测在[MACHINE_61_RESULTS.md](MACHINE_61_RESULTS.md)，不能当本机性能承诺。

### F. 用户自行启动正式训练

交付配置/脚本、显式预览开关、新结果目录与预估时长；按用户权限决定谁启动。
已有运行时不要改其源码/配方。完成2–3轮后用实际整轮时间更新ETA，验证、保存、启动都要计入。

## 3. 验收与交付清单

- [ ] 来源路径、数据计数/类别映射、转换命令、converter源码SHA256、manifest SHA256已记录。
- [ ] 原数据和旧结果保留，完整verify通过；无跳坏样本/重编码/数据污染。
- [ ] DDP步数、尾样本、worker覆盖、val计数、RNG、恢复语义已验证。
- [ ] 有效batch、模型、增强、优化器、精度保持固定；配置/源码有快照与哈希。
- [ ] 两后端可比短测、worker扫描、重复测、全量遍历结果和失败项均保留。
- [ ] 记录数据等待/step分位数、最慢rank吞吐、内存/换页、磁盘、GPU遥测。
- [ ] 明确缓存未控制、诊断额外同步、是否含验证/保存、收益比较口径。
- [ ] 正式启动须授权，不留自动训练调度器；没有代用户commit/push。

## 4. 若需要参考实际训练实现

原参考在独立训练仓库`spb_vit_dist/ablation/compute_matched_vit/`：
`tar_data.py`、`test_tar_data.py`、`benchmark_data.py`、`benchmark_pipeline.py`、
`run_tar_diagnostics.py`、`TAR_DATA.md`、`TAR_BENCHMARK.md`。
这些文件**没有随本库分发**，本库在没有该仓库时仍可转换/校验。
原8卡运行器还包含特定模型、checkpoint阶段和1251 updates/epoch检查，不能当通用脚本硬跑。
如无参考仓库，用本文档定义的契约适配自己的训练系统；不要猜不存在的CLI参数或忽略校验错误。
