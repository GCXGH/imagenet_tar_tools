# 后续测试计划：先定位瓶颈，再选择转换

## 方向更新（用户确认后）

目标已从当前服务器局部调优转向可复用ImageNet预训练数据层。
按顺序探索：WebDataset标准化原字节基线 → 解码/增强加速（DALI等） →
FFCV一体化管线 → 预缩放/编码策略。局部buffer/offset/绑核退居实现细节，
不再主导候选方案选择。下文保留为历史候选清单，不代表当前自动执行队列。

WebDataset阶段1已新增`webdataset_baseline/`：隔离小依赖、独立loader、
7项合成回归及128张真实图片原字节/PIL一致性检查通过。
当前仅正确性基线，不声称吞吐提升，尚无真实DDP或长测；详见该目录README。
未构造/下载大数据集，不自动进入下一种依赖安装或全量转换。

阶段2已做256图RAM处理/IPC诊断和单卡ViT-B/16正反序短测，
记录于`webdataset_baseline/PIPELINE_FINDINGS.md`。worker批量输出明显优于
逐样本IPC原型；uint8 loader更快却未改善该模型训练总吞吐。并未测全量TAR流式性能。
DALI只查询兼容性/包体积，未安装，停在大依赖下载前。

随后用户授权在spb安装DALI，并明确批准15分钟续传/安装及<=5分钟单卡小测。
DALI cuda120 2.4.0及13个新增distribution安装完成，不改变原torch等版本，pip check通过。
128图CPU/mixed decode、GPU确定性预处理及torch ViT训练功能通过；
mixed/PIL和resize/crop有输出差异，下一步先拆解，不发布加速结论。
见`dali/README.md`，未构造/下载数据集、未启动长测。

其后用户明确批准四卡batch1024完整epoch测试。original/pil_matched/dali三项串行完成，
同PBI模型/完整训练步，1251更新/项。DALI458.99秒 vs 原TAR624.94秒，
整轮吞吐+36.16%，首步后+33.35%；GPU平均util67.2%→82.6%。
配方实现、采样顺序和缓存不同，不能作精度/纯I/O结论。详见dali/EPOCH_RESULTS.md。
本轮任务结束，不自动接续反序重测、组件消融或精度训练。

用户随后批准解码消融/反向基线，三项完整epoch已完成。DALI CPU decode .33049秒/步，
mixed .34798，原TAR反向基线 .46038（均post-first）。GPU JPEG解码不是收益必要条件；
下一重点应为GPU增强/传输/流水线贡献，不优先继续调GPU decoder。
详见dali/DECODE_ABLATION.md；目前任务均已结束。

以下为2026-11-10初次整理时的计划；当时未运行新测试。
当前执行进展以上方“方向更新”和各方案README为准，未启动后台任务。
参考文档中的方案不是强制全量清单；以下按本机成本、潜在收益和配方风险排序。

## 1. 优先级与决策依据

| 优先级 | 方案/问题 | 首个验证 | 全量额外产物 | 决策门槛 |
|---|---|---|---|---|
| P0 | TAR剩余瓶颈分解 | 相同样本 encoded/decode/augment/loader/H2D | 无 | 磁盘/CPU/IPC/H2D/rank差异有证据 |
| P1 | TAR reader/worker/NUMA | 缓冲、workers、线程与调度的小范围对照 | 不需要重打包 | 精确计数+端到端收益，不只 loader快 |
| P1 | CPU管线/传输优化 | PIL vs torchvision解码；uint8 vs float batch | 无 | 解码/增强差异明确，IPC/H2D确有收益 |
| P2 | FFCV | 隔离环境可行性+少量Beton读写 | 需测，可能数十至数百GiB | 合法DDP顺序+有界RAM+配方可解释 |
| P2 | DALI | 用现有原字节TAR检查reader/decoder | 理想情况下无需图像副本 | 4090剩余显存可承受，不能只看loader |
| P3 | ZIP缓存 | ZIP_STORED/no与小预算part缓存 | 全量约原payload量级 | 93GiB不足以全缓存，先证明局部缓存价值 |
| P3 | Page cache/SSD | 观察缓存，确认真实可用存储 | 复制约原数据量级 | 没有可用SSD；禁止占满根盘 |

LMDB/WIDS/其他索引格式先不实施：在机械盘上逐样本随机访问仍可能受seek限制，
“换容器”不自动解决解码/增强；只有出现需要随机索引的实际需求再单独建目录。
WebDataset库接入也不是重打包理由，已有TAR格式可复用；如测试库本身，应独立目录并核对epoch语义。

## 2. 第一轮：无需新数据副本的诊断

后续先编写独立限额 runner，默认 dry-run；不导入生产 train.py，也不把真实训练增强
偷换成简单 resize。实际配方要显式记录，保留Pillow posterize零位兼容行为。

1. 从现有 JSONL 选择最多1,000条分层样本（类别/字节大小，若读取尺寸则记录），
   写相同 ID/label/hash manifest；不枚举全量原始目录，不读取完整shard来构建小样本。
2. 对比 source files 与现有 TAR 的原字节读取，随后 PIL RGB、训练增强、DataLoader。
   用相同样本和顺序，区分hash验证成本。tiny样本只做功能与局部剖析，不发布全量HDD加速倍数。
3. CPU上从0/2/4 workers起，记录线程数、队列等待、内存及磁盘竞争。
4. 单卡H2D检查先选择GPU4–7中一张确认空闲的卡；输出 dtype/pinning分别验证。
   pinning故障仍未明，不默认启用；若报错保留栈，不静默回退。
5. 如值得扩大，再申请四卡固定batch1024的实际训练步对照，覆盖相同卡组、样本和增强。

本轮不承诺该runner已有可运行命令；实现后提交准确命令和预算再进行扩大测试。

## 3. 第二轮：低风险优化

- TAR：现有4MiB缓冲基线，测试1/4/16MiB；比较成员头解析成本、起点扫描和边界多次打开。
  若需要offset index，优先只为小样本构建，不能悄悄遍历完整146GiB。
- 调度：四卡实际是32个worker并发读，可能打散机械盘顺序；比较较少I/O流+解码并发，
  不为顺序读取牺牲跨rank唯一性或尾批一致。
- NUMA：GPU4–7在node1，但只有28个本地物理核，32 workers加trainer可能过量。
  比较不绑核与分配CPU集合；不直接给所有rank同一个小集合，不随意强制membind造成局部内存不足。
- worker sweep：先固定prefetch1，再申请4/6/8/12；一次只改一个变量。
  shuffle512基线，256/1024作为采样与内存变化单列，不能混称纯I/O优化。
- CPU管线：float32输出每张约0.574MiB；uint8为其1/4，可减少IPC/H2D，
  但Normalize/RandomErasing迁移顺序与随机性需核对。先检查收益，再决定是否引入GPU算子。
- persistent_workers是候选而非直接开关；必须解决epoch/augmentation seed传播再测。

## 4. 第三轮：FFCV / DALI / ZIP

分别按子目录清单做兼容性、小样本正确性和资源估算。不要把三种安装/转换串成后台队列。
FFCV的jpg重编码/缩放属于新输入配方；DALI算子替换属于新预处理实现；
这两类与纯原字节TAR优化分表报告，不能仅用原始pipeline公开精度作公平对照。

## 5. 扩大执行前要提交的批准单

当前没有待自动执行的高开销命令。每次先给：

- 要回答的假设、备选配置、为什么小测不足；
- 精确命令、硬超时、CPU/RAM/GPU上限、同时运行的作业；
- 源/目标路径、预计读写量、产物和临时空间上限，根盘不能用作大数据目标；
- 成本估算来源和区间，磁盘负载/页缓存不确定性；
- 停止条件及回滚：终止自己的进程，保留诊断，只清理明确授权的新产物，不碰原始/TAR/训练结果；
- 单次批准后是否停止（默认是），不自动接续全量转换、长测或训练。

建议下一步先实现并检查第一轮的有限样本诊断工具，而不是立刻再做一种全量格式。
