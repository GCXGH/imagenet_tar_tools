# ImageNet 数据管线加速探索

目标：在现有 ImageNet 上减少训练端到端耗时，同时明确区分原字节 I/O、
采样顺序、解码/增强实现和图像内容的变化。不为了跑齐格式清单而全量重复转换。

## 目录边界

| 目录 | 负责范围 | 当前状态 |
|---|---|---|
| `tar_original/` | 已验证 TAR v1 的配方、结果和生产接口说明 | 已有全量数据、短测与 PBI 100e 长跑 |
| `webdataset_baseline/` | 标准WebDataset组件、独立训练接口与epoch策略 | 正确性通过；256图RAM处理/IPC与单卡ViT短测完成，未测全流式性能 |
| `tar_tuning/` | TAR 读取、缓冲、调度和 worker/NUMA 优化 | 已做有限prefix诊断；未替换生产reader |
| `cpu_pipeline/` | JPEG 解码、CPU 增强、输出 dtype、H2D | 计划 |
| `ffcv/` | Beton 转换、FFCV loader 与配方偏差 | 文档调查；未安装、未转换 |
| `dali/` | CPU/mixed JPEG 解码和 GPU 预处理 | 三组四卡batch1024完整epoch完成；DALI相对原TAR吞吐+36%，有配方/缓存限制 |
| `zip_cache/` | 无损 ZIP、Swin 风格缓存及 sampler | 已阅读官方代码；未转换 |
| `storage_cache/` | 页缓存观察、限额缓存、存储介质对照 | 计划；未预热、未复制 |

每个方案自己的 `src/`、`tests/`、`configs/`、小型结果记录在自己目录内按需创建。
初次整理只创建说明和现状记录；随后按用户要求增加限额prefix诊断工具和结果，
见 `tar_tuning/IO_FINDINGS.md`。不放不能执行的占位benchmark，不预装所有方案依赖。
不跨方案导入实现，不导入 `spb_vit_dist`；将来需要相同输入时使用相同样本 manifest，
不是共享或修改生产 trainer。各方案可有独立环境和依赖，不使用一个大而全 requirements。

旧 TAR 转换器保留根目录原位，这是为已发布数据保留的兼容例外，不是其他方案的基类。
生产 TAR reader 目前仍在训练仓库；这里明确记录来源，不把含模型/训练逻辑的代码复制进工具库。
未来若抽取独立 reader，要做行为对照、独立版本化，不反向改动现有正式实验。

大型产物仅在批准后放到 `/data1/xinghaitong/dataset/<方案独立目录>/`。
原始目录、现有 TAR 和结果均不得覆盖。仓库只保存代码、配方、小型报告及可追溯元数据。

## 阅读顺序

1. [现有 TAR](tar_original/README.md) 与 [实测表](tar_original/results.csv)
2. [本机资源快照](ENVIRONMENT.md)
3. [后续测试计划](ROADMAP.md)
4. [统一测试协议和授权规则](PROTOCOL.md)

所有“计划”均不代表已经启动后台任务。下载大型资产、全量转换、长时间测试和训练
必须针对具体命令再次得到用户批准；历史批准不自动延续到新方案。
