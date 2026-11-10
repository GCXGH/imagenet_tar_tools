# TAR 是否消除了 I/O 瓶颈？阶段性确认

2026-11-10。结论：**没有证据证明已经消除，并已确认一项可避免的读取放大。**
格式收益不等于reader、调度、缓存与物理存储已经最优。

## 1. 本次操作边界

- 不改生产代码/数据、不用GPU、不安装依赖、不清缓存、不启动后台任务。
- 仅读一个训练shard的sidecar前832条元数据，选第768起的64张图片，
  baseline/indexed/indexed/baseline共4次；raw文件接口读取总计192MiB，约0.33秒测量时间。
- 工具有45秒报警时限、256MiB raw-read预算；默认dry-run，显式execute才读数据。
- 一个小型合成USTAR单测通过，验证两种路径hash/order相同、读取放大减少、越界/预算拒绝。

命令（已执行的有限诊断；结果路径已有文件，重复会拒绝覆盖）：

```bash
python experiments/tar_tuning/probe_prefix_io.py \
  --root /data1/xinghaitong/dataset/imagenet1k_original_tar_v1 \
  --execute --result experiments/tar_tuning/prefix_probe_20261110.json
```

## 2. 已确认：分片起点prefix扫描的额外读取

生产reader用4MiB BufferedReader和tarfile `r:`，每段从archive开头解析成员到start。
虽然extractfile只读取目标样本，但成员间跳转与缓冲交互会把prefix payload读入用户缓冲。
因此“没有显式读取被跳过的图片”不等于“没有额外I/O”。

利用v1 USTAR和sidecar里的payload字节数，可累计header/512字节padding得到目标起点。
探针只为一个小窗口计算offset，再在该位置开始tarfile解析；不改变后续顺序读取，
不逐图片随机seek，也没有生成全量持久化索引。

同64张图片，原payload共7,174,431 bytes（约6.84MiB），所有hash/label/order一致：

| 路径 | raw文件接口读取 | read系统调用数（包装层） | raw seek次数 | 第一次/第二次耗时 |
|---|---:|---:|---:|---|
| 原prefix walk | 88MiB | 22 | 18 | 0.207 / 0.072秒 |
| 定位起点后顺序读 | 8MiB | 2 | 1 | 0.023 / 0.021秒 |

文件接口读取减少约90.9%，局部读取放大从12.86降到1.17。
**这不是整轮11倍加速，也不是严格冷缓存A/B。**
首次baseline的/proc/self/io read_bytes约63.13MiB，随后indexed约0.625MiB、其余0；
indexed受前面baseline暖缓存影响。此指标是内核为该进程计入的storage读取，
不是直接测量物理盘磁头流量。缓存可能隐藏物理读，但CPU拷贝/解析开销仍存在。
时间还包含payload SHA256校验，不能当作生产loader吞吐。

计划级估算（只读取manifest、模拟100个epoch，无图片遍历）：
world4、batch256、workers8，平均每轮32个非零prefix起点；
prefix额外遍历样本头平均65,223（59,244–70,860），约保留样本数的5.09%。
这是样本头数量，不是物理字节量。说明该修复有价值但范围有限，
必须对整个epoch衡量，不能把小窗口比值外推。

## 3. 仍未确认的磁盘侧问题

1. 32 workers跨shard并发读取是否把顺序I/O变成多个竞争流，产生额外seek与尾延迟。
2. 每epoch真实物理读取量、read amplification、缓存命中与readahead浪费。
3. rank/worker到达shard边界的长尾是否主导最慢rank。
4. fewer I/O streams + independent decode workers能否兼顾带宽和CPU并行。

没有采集旧100e期间的逐时iostat/各rank数据，不能事后宣称磁盘已经饱和或完全空闲。
本次3秒空载检查sdb较空闲，也不能代表训练运行时。

## 4. 为什么12.8%不能排除I/O瓶颈

data_wait记录主线程从上次step提交后到取得下批的CPU时间，只在rank0汇总。
它混合磁盘、decode/augment、IPC/调度和epoch worker启动；取batch时GPU可能仍执行前一步。
它不含H2D及其他rank等待；其他rank读得慢也可在DDP同步上体现。
因此79.3秒/轮不能都归为磁盘，也不能作为所有I/O可优化收益的上限。

完整长跑分段可见等待稳定存在：第2–20轮13.21%、21–40轮13.33%、
41–60轮13.39%、61–80轮12.49%、81–100轮11.72%。
最后一段训练反而较慢而rank0等待减少，也说明单一rank指标不足以解释全部步时。

## 5. 下一项确认实验（未执行）

优先在独立reader里实现offset起点版本及正确性对照，不重打包数据；
随后申请有时限的中等规模对照，固定卡组/样本/增强/缓存标注，
比较原reader/offset reader，以及少量worker并发档位，记录磁盘和最慢rank。
不要一次同时修改reader/worker/shuffle/decoder。

要判断“训练仍受多少磁盘限制”，还需要相同原字节计划的read-only、
decode/augment loader和真实train-step三层测量；局部探针已经确认可优化点，
尚未确认全量端到端收益。长时间或多卡测试先提交准确预算和命令，等待批准。
