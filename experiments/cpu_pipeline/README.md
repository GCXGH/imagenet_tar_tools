# 解码、增强、IPC 与 H2D（尚未实施）

先在同一组原始字节上分解PIL decode、RGB转换、RandomResizedCrop、RandAugment、
ToTensor/Normalize/RandomErasing、batch组装、IPC和H2D。
候选是 torchvision CPU JPEG、其他隔离decoder、uint8 batch与GPU normalize，
不是一开始替换所有增强。测试自己的数据适配器，不跨方案import，不改生产trainer。

检查灰度/CMYK/非JPEG、坏图处理、颜色/插值/像素差异、resize/crop参数和RNG传播。
某decoder只支持JPEG时必须显式记录格式覆盖及fallback计数，不能静默过滤样本。
从CPU单进程、2/4 workers到单卡短测；全量或多卡再审批。

当前float32 `[256,3,224,224]`约147MiB；uint8约36.75MiB。
32 workers×prefetch1的在途tensor规模不可忽略，但不是可直接推算的峰值RSS，
要计入队列、共享内存、trainer持有batch及decoder临时空间。
Normalize迁移应在float中进行，RandomErasing原位置和Mixup/CutMix输入语义保持可追溯。
增强或decoder变化要与纯I/O结果分开。

成功门槛：正确性、CPU/RAM及H2D收益，同时不因GPU算子/显存争用拖慢真实train-step。
当前没有新增依赖、真实图像测速或GPU任务。
