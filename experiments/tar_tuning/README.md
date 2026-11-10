# TAR 调优（独立实验，已有有限 prefix 诊断）

假设：原TAR已解决大量小文件open，但32个worker并发读、Python成员解析、
prefix扫描、IPC与NUMA可能仍有限制。无需再造一份全量数据。

输入只读复用现有TAR与sidecar。新reader代码/测试放本目录，不导入生产训练仓库，
不改根目录转换器，也不要求其他候选方案依赖它。

逐项比较：文件buffer 1/4/16MiB、tarfile解析/offset读取、I/O流数量与解码并发拆分、
workers、线程限制、CPU affinity；最后才动prefetch/shuffle/epoch worker生命周期。
索引生成也是I/O处理，完整扫描须申请；mmap不等于缓存常驻，也不一定适合HDD随机读。

验收：与冻结基线逐sample的ID/label/payload hash一致，train尾部/全局batch数、
val完整覆盖、epoch确定性/恢复、多worker和DDP步数一致。
吞吐评价必须包含真实增强和实际训练，诊断与低插桩计时分开。
严格遵循上级 PROTOCOL。当前只有独立的限额 prefix probe，未替换生产reader、
未构建全量索引、未运行长测。结果见 [IO_FINDINGS.md](IO_FINDINGS.md)。
