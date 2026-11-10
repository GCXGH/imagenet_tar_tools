# 存储介质和缓存（只调查，未复制/预热）

页缓存不是新格式。当前全量约145GiB TAR archive，机器93GiB RAM不能全装下；
小pilot可热缓存，不代表全量HDD效果。只观察/proc/meminfo、进程PSS/RSS和磁盘I/O，
不drop_caches、不vmtouch锁页、不建大型tmpfs。

可候选：固定预算的压缩字节缓存、验证集局部缓存、已有空闲SSD上的同样本副本。
限额缓存须量aggregate内存、命中率、首轮成本和epoch采样变化；缓存旧随机增强输出
会改变训练数据，不能以decoded/transformed tensor缓存代替无损输入缓存。

本次lsblk未发现NVMe/可用SSD挂载，sda/sdb均报ROTA=1；根盘仅约15GiB剩余，
因此SSD全量副本暂阻塞，不为此采购/下载/占用云盘。
数据目录位于同一/data1时换路径不是换介质；同盘复制可能干扰其他作业且无存储隔离收益。

全量预热/复制需精确源目标、读写量、空间余量、时间与回滚方案，并先得到批准。
