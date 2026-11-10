# 安装记录（2026-11-10）

检查spb：nvidia.dali不可发现，pip/conda无DALI distribution；已有nvidia-*是torch CUDA依赖。
torch2.7.1+cu118、torchvision0.22.1+cu118、numpy2.5.4、packaging26.3。
候选DALI cuda120 2.4.0，pip dry-run仅新增13包，不替换已有依赖，总wheel约498.90MiB。
首次240秒pip下载超时，未安装，留下约181MiB主wheel片段。
用户随后明确批准最多15分钟续传/安装，之后<=5分钟单卡小测；失败停止。
安装与下载不代表数据转换授权。无数据集下载/新副本。
