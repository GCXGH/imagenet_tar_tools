# ZIP 与 Swin 风格缓存（已读官方代码，未转换）

原字节ZIP_STORED可降低小文件open，但随机ZIP成员访问仍可能seek；
不假定ZIP比现有TAR更快，也不对已压缩JPEG默认使用DEFLATE。
用极小ZIP先验证ID/label/hash及no/part缓存，再决定是否值得全量转换。
实现/测试仅在本目录；若复制上游代码，必须固定commit并保留其许可证，而非只留网址。

2026-11-10阅读官方main分支（非固定commit，尚未vendoring）：
- https://github.com/microsoft/Swin-Transformer/blob/main/data/cached_image_folder.py
- https://github.com/microsoft/Swin-Transformer/blob/main/data/build.py

已核对行为：`cache_mode=part`按 `index % world_size == rank`缓存压缩字节；
build_loader同时使用rank固定子集的SubsetRandomSampler。不是标准DistributedSampler
每epoch重新划分rank的语义。`full`可能在每rank复制整份，不是共享一次。
不能只设置cache_mode而不核对实际sampler，也不能用原验证loader的padding行为替代完整覆盖要求。

本机总RAM约93GiB，小于全量原图payload量级；part只是分摊，总RAM并不自动除以GPU数。
加上page cache、Python对象、workers和训练内存，全量缓存不作为默认候选。
可评估严格限额缓存，但要记录命中率和预热成本；原版part与限额缓存是不同实验。
全量ZIP和长期缓存预热均暂停，待小测有价值后申请。
