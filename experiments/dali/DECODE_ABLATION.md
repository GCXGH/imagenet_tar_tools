# 解码路径消融与反向基线（2026-11-10）

## 完成结果

18:28–18:54三项全部完成，所有rank1251步、320256个输入，梯度存在且有限。
生产模型/train/reader源码hash与运行前一致；GPU4–7已释放，无后续任务。

| 管线 | epoch秒数（含首步） | images/s | 首步后秒/步 | 平均GPU util | 单卡总显存最大采样 |
|---|---:|---:|---:|---:|---:|
| DALI mixed decode + GPU增强 | 464.77 | 2756.3 | .34798 | 85.2% | 21.69GiB |
| DALI CPU decode + GPU增强 | 442.34 | 2896.0 | .33049 | 85.0% | 20.62GiB |
| 原TAR + 原配方（最后运行） | 619.10 | 2069.2 | .46038 | 67.2% | 19.95GiB |

CPU相对mixed：整轮吞吐+5.07%，首步后+5.29%，总显存少约1.08GiB/卡。
CPU DALI相对原TAR：整轮吞吐+39.96%，时间减少28.55%，首步后吞吐+39.30%。
mixed相对原TAR：整轮吞吐+33.21%，首步后+32.30%。
以上仍是单次解码对照，不能把5%差距当作跨机器/模型稳定收益。

## 能得出的归因结论

- GPU参与JPEG解码**不是本次DALI收益的必要条件**。保持GPU增强/后处理时，
  CPU解码路径仍保持整套加速，并在本轮更快、占显存更少。
- 不能因此断言CPU解码“贡献0”或GPU增强“贡献40%”：DALI CPU路径也改变了
  reader、原生线程调度、random-crop decode、数据表示/传输及流水线。
- mixed的GPU util85.2%与CPU的85.0%几乎一样，但CPU更快，进一步说明util不能替代吞吐。
- 可能的解释是CPU已能供给足够解码结果，mixed增加GPU资源/同步压力，
  但没有GPU timeline/CPU profile，尚未验证具体机制。

原TAR反向基线.460秒/步，与上一轮.457接近；mixed本次.348与上一轮.343接近。
这削弱了“DALI只是最后运行缓存更热”的解释，但不排除样本顺序、缓存和其他作业差异。
CPU/mixed只跑了一种先后顺序，CPU所读设备字节114.14GiB vs mixed112.94GiB，
不能据此说缓存严格相同。这是全设备计数，不是进程归因。

## 下一项优先级

优先构建DALI CPU decode + CPU augment的中间组合，与CPU decode + GPU augment对照，
再拆float/uint8传输和输出布局。保持reader/输入计划，先在小样本核对算子和标签。
即使CPU增强慢，也不能把差值全归“GPU增强算力”：跨设备传输位置和表示也会一起变化，
需要明确列出数据流，并以matched组合或时间线进一步归因。
不再优先优化GPU JPEG decode；暂推荐CPU decode作为该机器/该模型后续消融基线，
不直接替换正式实验，不将单轮吞吐当作精度/收敛验证。
尚未进行全epoch ID唯一性审计、CPU/GPU增强消融、预生成GPU batch重放或新精度训练。

以下为本轮执行前设计与预算记录。

用户同意继续归因实验。本轮三个完整epoch串行：
1. dali_mixed：DALI reader + mixed random-crop decoder + GPU resize/flip/RA/normalize。
2. dali_cpu：同reader/参数/线程/seed，换CPU random-crop decoder，之后GPU处理不变。
3. original_reverse：原TAR+原timm配方，在DALI后运行，减轻上次固定顺序的解释歧义。

GPU4–7、batch4×256=1024、PBI wider、BF16/compile default、AdamW/Mixup/CutMix/clip，
每项1251更新。每项30分钟硬时限，预计共25–45分钟；不下载数据/安装包/生成新索引。
CPU和mixed可能走不同JPEG实现，故这是解码路径的系统消融，不是像素等价训练轨迹。
CPU decoder并未把之后的增强搬回CPU，不能误称为“CPU增强 vs GPU增强”。

CPU路径四卡12步预检通过，梯度有限，post-first约.345秒/步。
这与mixed旧短测接近，但等完整epoch才作结论。

入口：`run_epoch_suite.py --suite decode_ablation --output <新目录> --execute`。
结果根：`/data1/xinghaitong/results/imagenet_io_benchmarks/dali_decode_ablation_20261110/`。
调度日志：`/tmp/opencode/dali_decode_ablation_20261110.log`。
每2秒监控GPU/总显存/主机可用内存/设备diskstats；只读分析`analyze_epoch_suite.py`适用于两组suite。
采样顺序和缓存不能严格受控，不清全系统缓存；反向基线也不是严格冷缓存试验。

本轮尚不能独立量化GPU增强、uint8/float传输和原生reader各自贡献。
这些需要额外中间管线，不用CPU/mixed对照代替。完成后依据结果选择下一项，不自动接续长测。
