# FFCV（候选方案，未安装/未转换）

独立环境、独立src/tests/configs/results；不升级spb，不让其他方案依赖FFCV。
本机Python3.12、torch2.7.1+cu118的可行安装组合尚未验证，先调查FFCV/numba/
compiler/libjpeg-turbo/OpenCV/CUDA约束；不要预判可用，也不要因复杂而无证据略过。
创建大环境、下载大依赖前报告位置、下载量和磁盘预算（根盘仅约15GiB剩余）。

## 首个实验

最多1,000张已有图片，显式 `RGBImageField(write_mode='jpg', max_resolution=512,
jpeg_quality=90)`，再比较不预缩放或1600/Q90（仅在有价值时）。
这些会重编码，512也可能缩小图像，不属于原字节I/O-only。
记录实际API、decoder、输出dtype/device、augmentation顺序及label/sample ID。
raw/smart的体积先估算，不做全量raw。

重点先验证：

- 具体安装版本的write_mode/API默认值，不依赖文档示例推断实际行为。
- `distributed=True` 支持哪些order。官方文档提示不支持QUASI_RANDOM；
  对大于RAM的数据，盲目改成RANDOM也可能造成严重内存/随机I/O问题。
  因此单卡QUASI_RANDOM表现好不能自动外推四卡可行。
- os_cache=True/False、order对aggregate内存的影响；mmap不是整份常驻。
- DDP覆盖、尾部与每rank步数；若合法DDP顺序不适合本机内存，要准确报告阻塞，
  不能悄悄固定rank子集并称等价。
- RandomResizedCrop decoder、插值、RandAugment/Erasing等与原配方的差异。

全量Beton大小与转换时长由代表性小样本外推区间，不用公开数据假定本机文件大小。
完成小测后停止，提交全量输入/输出/临时空间/超时/占用及准确命令，再等批准。

2026-11-10已阅读官方文档：
- https://docs.ffcv.io/parameter_tuning.html
- https://docs.ffcv.io/working_with_images.html

尚未安装实测，无性能或精度结论。
