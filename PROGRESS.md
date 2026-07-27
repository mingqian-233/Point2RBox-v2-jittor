# PROGRESS（追加式日报）

## 2026-07-26（Day 1）

### 做完什么
- **M0 完成**：两套 conda env 实测就绪并通过验收
  - `p2r-jittor`：py3.10 + jittor 1.3.8.5 + numpy 1.26.4，`cc_path=g++-10`；
    `test_example` exit 0、GPU 1000×1000 matmul 误差 2.1e-4、cudnn conv 可用
  - `p2r-torch`：py3.12 + torch 2.2.0+cu121 + mmengine 0.10.7 / mmcv 2.2.0 / mmdet 3.3.0 +
    mmrotate 1.0.0rc1(editable)，版本自检与上游 environment.md 逐项一致；
    官方 `train.py` 冒烟通过（停在数据缺失，符合预期）
  - 产物 `scripts/setup_env.sh` + `docs/environment.md` 已 push；COORD 已通知 B
- **M1 进行中**：底座 clone + remote 配置 + `setup.py develop` 完成；
  基线 `point2rbox_obb_r50_adamw_fpn_1x_dota.py --task=train` 跑到 dataset 构建、
  报错为"找不到数据"（上游写死路径 `/home/yi/Data/processed_DOTA/...`）而非代码错 → **底座是活的**
- README 顶部加 fork 归属声明

### 关键发现 / 偏差
- `[plan-deviation]` **numpy 2.x 会让 jittor 1.3.8.5 静默产出垃圾数值**（无任何报错，
  `jt.array` 喂进算子输出未初始化内存）。计划未预料。两个 env 都钉死 numpy==1.26.4。
  ⚠️ `setup.py develop` 装 jdet 时 numpy 又被顶回 2.2.6，已重新钉回——**以后每次 pip 后必查**。
- `[plan-deviation]` 计划预案的 g++-11 实测不行（glibc 2.34+ 的带参 `__malloc__` 属性），
  g++-13 也不行（`__builtin_dynamic_object_size`），最终 **g++-10**。详见 docs/environment.md 坑 #2。
- JDet 底座的 DOTA 数据路径是「preprocess.py 切图 + labels.pkl」体系，与 mmrotate 的
  `split_ss_dota` 目录不同 → M2/C4 时要做适配（B 产出的是 mmrotate 风格切图）。
- B 问的 `RoIAlignRotated`：底座有 CUDA 实现（`ops/roi_align_rotated.py:312`），
  签名 `(output_size, spatial_scale, sampling_ratio)`，**无 mmcv 的 `clockwise` 参数**，
  角度约定待 L1 parity 验证（M2.5）。

### 卡在哪
- 无阻塞。

### 明天做什么
- ~~M1 收尾~~（当天已完成：derive_scope.py + port_scope.md 已产出）
- M2 开工：C1 梯度裁剪 / C2 组合 LR 调度（B 依赖，优先）
- L0 config parity 测试骨架

### 当前数字
- parity：尚未开始（M2 起有 L0/L1 数字）
- mAP：无

---

## 2026-07-26（Day 1，追加：M2 上半场）

### 做完什么
- **C1**：`AdamW` 支持 `grad_clip(max_norm, norm_type)`，pre_step 触发全局 L2 裁剪；
  单测：已知梯度（L2≈547.7）裁剪后范数=35.0（误差<1e-3）✅
- **C2**：新增 `LinearWarmupMultiStepLR`，与 mmengine `LinearLR×MultiStepLR` **1440 点逐点相等**
  （rtol 1e-9）。两个非直觉发现：
  1. mmengine `LinearParamScheduler` 分母是 `end-begin-1`（≠torch LinearLR），第 499 iter 到顶
  2. 底座 runner 原本 optimizer.step **之后**才 scheduler.step（实际生效 f(i-1)，滞后 1 iter）
     → 已挪到之前，对齐 mmcv/mmengine 语义。`[plan-deviation]`：这改变了底座所有 config 的
     LR 时序（1 iter 级，方向是修正），已在 COORD 通知 B
- **C3**：`Runner.train` 每 epoch 开始调 `model.set_epoch(epoch)`（等价 SetEpochInfoHook）
- **L0 骨架**：`tools/dump_config.py` 展平 3 个官方 config（296/297/246 键）进 golden；
  `test_L0_config.py` 铁律二锚定测试全过；Jittor config 侧比对留 M4 挂钩
- 测试面板：**7 passed**（L1×4 + smoke×3）+ L0 3 passed 1 skipped(预期)

### 剩余（M2 下半场）
- C4：DOTADataset 读 COCO `*.bbox.json` + `DOTAMetric(format_only=True)` 输出与 mmrotate 一致
  （等 B 的 split_ss_dota 校验通过，B 已切完 trainval 21046 / test 10833）
- M2.5 复用件核对（gaussian_dist_loss / PSCCoder / nms_rotated / RoIAlignRotated 角度约定…）

### 当前数字
- L1 LR 序列 parity：max 相对误差 < 1e-9（1440 点）
- C1 裁剪范数：|35.0000 - 35| < 1e-3

---

## 2026-07-26（Day 1，追加：实验 X 开跑 + M2.5 完成）

### 做完什么
- **实验 X（PyTorch 官方 baseline）在 GPU 0 开跑**（12:53 起，B 的 split_ss_dota 校验通过后立即启动）
  - 6400 iter/epoch（与 B 对账数一致），iter50 lr=1.994e-5（吻合 warmup 公式），
    loss_bbox_edg=0（epoch 6 才启动，符合预期），ETA ~9h。监控已挂（NaN/崩溃/epoch 边界）
  - 踩坑：需 `third_parties` symlink（ted.pth 相对 cwd 加载）+ mobile_sam.pt 完整版
- **M2.5 复用件核对全部完成**，测试面板 20 passed / 1 skipped：
  - GDLoss：修正底座 3 处与 mmrotate 的语义差异（execute 的 mask过滤→weighted_loss 语义、
    gwd det clamp 0→1e-7、reduce mean）；前向/梯度/加权路径 parity 全过
  - PSCCoder：修底座 `axis=` 残留（torch 风格，jittor 需 `dim=`）；encode/decode 过
    （decode ±π/2 端点 atan2 符号翻转，le90 下等价朝向，按模 π 角距离比较）
  - box_iou_rotated / nms_rotated：与 mmcv 一致。**已知差异**：jdet nms keep 原始顺序 vs
    mmcv 分数降序（M4 head 层要 sort 后截断 max_per_img）
  - ROIAlignRotated：加 mmcv 语义 aligned/clockwise 参数。**实测 JDet 原生方向 =
    mmcv clockwise=True**；EdgeLoss 用法 = (49, scale, ratio, aligned=True, clockwise=True)
- COORD 已回复 B（RoIAlignRotated 结论 + nms 顺序差异警告）

### 明天做什么（实际今天继续）
- M3：ops/linalg2x2.py（2×2 eigh/solve 闭式解，w==h eps 处理）→ 四个 loss 逐个移植
- M3 完成打 tag core-stable（计划 §3 与 M4 验收表述不一致，按 §3 的 M3 时点执行，
  B 正在等 tag 开工他的 M5）
- C4（DOTA COCO json + DOTAMetric）排在 M3 后（只有我的 stage-2 依赖，B 不等它）

---

## 2026-07-26（Day 1，追加：M3 完成，core-stable 发布）

### 做完什么
- **M3 全部完成，tag `core-stable` 已发布**（commit 9a6c57b），B 已在 merge
- `ops/linalg2x2.py`：2×2 eigh/solve/diag_embed 闭式解，16 用例（含 w==h/近各向同性/
  面积 1e-6/大条件数）6/6 过。关键实现点：EPS=1e-24（1e-12 会污染小面积框）、
  b 对称化读取（梯度在 [0,1]/[1,0] 对称分摊）、|delta|/disc≤1 保证梯度有界
- 四个 loss 逐个移植+parity+commit（源=v3 版，扩展参数默认=v2 行为）：
  - GaussianOverlapLoss：前向 rel<1e-4，mu/sigma 梯度 rel<1e-3（官方参数与 lamb 两路径）
  - VoronoiWatershedLoss：**watershed markers 与 PyTorch 逐像素一致**，前向 rel 3e-7
  - EdgeLoss：新移植 RotatedSingleRoIExtractor；前向 rel<1e-3
  - ConsistencyLoss：rot/flp/sca 三路径全过
- 测试面板：**34 passed / 1 skipped**（底座自带 test_dataset 的 2 个收集错误是其要数据文件，与本仓库无关）

### 新踩坑（已回馈给 B）
- jt reduce（.max()）出 shape [1]，stack 两个会得 (2,1) → 用 concat
- torch.meshgrid(indexing='xy') jittor 无 → 手工构造
- 简并 eigh 的逐特征值梯度是子梯度（与 torch 的差异只在基底分配，trace 一致）

### 训练巡检（实验 X @GPU 0）
- 13:28 epoch1 5350/6400：loss 3.94→2.47 稳降、lr 满 5e-5、grad_norm≈32（未触 35 裁剪线）、
  显存 5.3GB、0.38s/iter、ETA 7.5h。loss_bbox_edg=0 待 epoch6 符合预期

### 接下来
- M4：v2 head（43KB，按方法块拆 commit）+ detector + config（targets/bids 约定定下后同步 B）
- C4（DOTA COCO json + DOTAMetric）与 M4 交错做

---

## 2026-07-26（Day 1，追加：M4 主体完成）

### 做完什么
- **TED 移植**（third_parties/ted，权重转 pkl，4 输出 parity 过）——v2 detector 依赖，
  放 third_parties 保持上游 import 路径且不越 B 的 models/edge/* 命名空间
- **detector 移植**（networks/point2rbox_v2.py）：dual-stream 三路增广 + bids 维护 +
  copy-paste cache + TED 边缘注入。targets 约定已在 COORD 公布（B 照抄）
- **head 移植**（roi_heads/point2rbox_v2_head.py，922 行源）：
  - index_reduce_('amin') 语义分析为组内代表（unique 组内恒同值）；'mean'（带梯度）
    用 one-hot 矩阵乘；unique 走 numpy（detach 量）
  - GPU 冒烟：predict + 6 losses + 双分支梯度非零
- **GPU 阻塞点修复**：jt.linalg.det/inv 是 numpy_code（GPU 要 cupy）→ 全部换 2×2 手工式；
  cublas_batched_matmul 不广播 → solve_2x2 显式 expand
- **DistanceAnglePointCoder** 移植（boxes/coder.py）
- **C4a 数据集**（data/p2rv2_dota.py）：直读 B 的 split_ss_dota txt。三个不能复用底座的点：
  point_dummy 官方=1（底座 0.1）、RotatedResize 会重规范角度（点框 0→-π/2，破坏 head
  约定）→ 自写 MMRotateResize、RotatedRandomFlip 翻转数学与 mmrotate 不同 → 自写
  MMRotateRandomFlip。**filter_empty_gt 后 12800 样本 = B 对账数 = 官方日志 6400 iter@bs2** ✅
- **config**（configs/point2rbox_v2/point2rbox_v2_1x_dota.py）逐行对照官方 +
  **L0 Jittor 侧逐值比对 10 passed**
- 冒烟训练已启动（与实验 X 共卡，首跑 JIT 编译中）

### 当前数字
- 测试面板：45 passed / 1 skipped（L0×10 + L1×15 + L2×14 + smoke×3 + 其余）
- 实验 X @GPU0：epoch 2，loss 2.05 稳降，ETA ~7h

---

## 2026-07-26（Day 1，追加：M4 验收达标）

### 做完什么
- **TestHeadParity 通过**：固定权重（torch state_dict 直传，键名完全一致）+ 固定输入下，
  6 项 loss_dict 全部 rel<1e-3、feat 梯度对齐 —— M4 数值验收达标
- 揪出一个**必然导致训练失败的大 bug**：底座 FocalLoss 是 1-based 标签约定
  （bg=15 会被映射成第 14 类正样本），首次冒烟 loss_cls=2306 暴露 →
  新写 MMDetFocalLoss（对齐 mmdet py_sigmoid_focal_loss 语义）
- 修 ConsistencyLoss rot 分支的 GPU matmul 广播（冒烟暴露，CPU 测试测不到）
- docs/config_parity.md + docs/porting_notes.md 交付文档入库
- 首次冒烟：跑到 iter 100 无 NaN、warmup LR 曲线正确（2.0e-5→2.33e-5 吻合公式）；
  已用修复后代码重启验证
- ⚠️ 速度：首跑 0.4-0.5 fps（与 torch baseline 共卡 + JIT 编译期），折 2-2.5s/iter vs
  torch 0.38s/iter——待 baseline 让出 GPU 后复测，若仍慢即触发 R5 缓解（bucket padding）

### 测试面板
- **43 passed / 1 skipped**（L0×10 + L1×15 + L2×15 + smoke×3）

---

## 2026-07-26（Day 1，追加：M4 收官 + M6 正式训练开跑）

### 关键事件
- **又修一个底座致命 bug**：`ops/nms_rotated.py` CUDA kernel 的宿主后处理直接解引用
  device 指针，默认 allocator 下任意 N 段错误（只在 managed allocator 下侥幸可用）。
  改 mmcv 标准显式 memcpy，N=16~5000 全过、CPU/GPU keep 集合一致。**该 bug 会炸掉
  所有推理/评测路径**，已通知 B（他的 v3 推理也会踩）
- 测试路径全通：score_thr 0.001 → 1577 dets → 逐类 NMS → 降序 → max_per_img cap
- **冒烟即达 M4 验收**：iter 1350 无 NaN、warmup 正确、loss 量级与 torch 同档
  （cls 0.6-0.7 / bbox 0.8 / vor 1.0-1.3 / ovl 0.7-1.1）
- **R5 风险解除**：JIT 暖机后 2.6 fps（0.38s/iter，与 torch baseline 相当，且是共卡实测），
  变长 GT 未触发重编译灾难，无需 bucket padding
- **M6 正式训练启动**（14:25，commit 4c15ae7 干净状态，logs/train_v2_official.log，
  ETA ~16h 共卡 / baseline 21:00 让卡后会加速）。监控挂 epoch 边界/NaN/mAP

### 测试面板
- 43 passed / 1 skipped（head feat 梯度改稳健度量：整体相对 L2<1e-3 + 违约<0.2%，
  GPU conv 反向原子累加有 ~0.1% 非确定尾部）

---

## 2026-07-26（Day 1，追加：训练死锁处置 + M7 准备）

### 事件：jittor dataloader 死锁（15:05）
- 正式训练 40min 后卡死：py-spy 确诊 worker 卡 `buffer.send`（环形缓冲满）、
  主进程卡 `idqueue.pop`（等 worker 信号）——jittor 多进程 dataset 的竞态死锁
- `[plan-deviation]` num_workers 2→0（加载性能参数，不动任何数值语义）；
  15:14 重启（该 run 为正式 M6 run），实测 fps 2.0（慢 ~15%），ETA ~21h（共卡），
  已跑过原死锁点无复发
- ⚠️ 遗留：跑着的进程加载的是 execute 分发修复前的代码，epoch 12 的 val 会
  走错分支——届时 kill 前确保 ckpt_12 已存，用新代码离线跑 val/test

### M7 准备完成
- detector.execute 分发修复（eval 带 GT targets 误入 forward_train）
- C4a 完成：P2RV2DOTADataset 支持 ann_json（COCO bbox.json 读取，往返测试过）
- tools/generate_pseudo_labels.py：输出与 mmrotate DOTAMetric.results2json 逐字段一致
- 发现：底座 FCOSHead 已是带角度分支的 rotated FCOS → stage-2 剩余工作 =
  语义核对 + RotatedIoULoss（可微旋转 IoU 移植）+ config

### 巡检（15:40）
- 实验 X：epoch 3/12 完，loss 1.99 稳降（cls 0.35 / vor 0.92 / ss 0.22），ETA ~23:00
- Jittor v2：重启后正常，loss 量级同档（cls 0.54 / vor 1.17），偶发 ss 尖峰有 clip 兜底

---

## 2026-07-26（Day 1，追加：训练稳定确认 + L3 达成 + RotatedIoULoss）

### 训练线
- **Jittor v2 第 5 次启动（17:20，commit ff010a3）后稳定**：iter 4000+ 跨过历史卡死点，
  fps 2.84（两处对数化修复后反而更快），ETA ~14h。第二处卡点也已根治
  （L_target 逐实例 np.nonzero O(J·HW) → argsort 分桶 O(HW log HW)）
- 实验 X：epoch 6/12 完，loss 1.82 稳降（edge loss 预计 epoch7 激活，届时核对）

### M5/M7 硬件完成
- **RotatedIoULoss / diff_iou_rotated_2d 移植 + parity**：IoU 前向 max diff 5.6e-6、
  loss 与 mmcv 逐位一致、非退化梯度 rel 3e-5（重合框的重复顶点子梯度 mmcv 亦任意）。
  关键坑：排序在归一化坐标、面积必须在原始坐标（填充槽零贡献依赖原始系）
- **L3 达成（静态强形式）**：tools/convert_torch_ckpt.py（键 1:1，0 缺 0 多）+
  实验 X epoch_6 真实权重整模型前向 parity——cls/bbox/angle 相对 L2
  1.2e-4 / 6.7e-4 / 1.2e-3。权重转换与全链路数值一次性验证
- 测试面板 45 passed / 1 skipped（新增 L3）

### 当前数字
- 全模型前向（真实权重）：cls 1.2e-4 / bbox 6.7e-4 / angle 1.2e-3（相对 L2）

---

## 2026-07-26（Day 1，追加：事故与规矩）

### ⚠️ 事故：误杀 Agent B 的 v3 训练（18:44，全责在我）
- 清理自己多代重启的孤儿进程时，对一个持有 GPU 的 python 进程未核对命令行就 kill -9，
  事后发现是 B 的 v3 训练（b_v3_train_entry.py）。已在 COORD 如实报告并致歉，
  B 需从最近 checkpoint resume
- **立规**：任何 kill 之前必须 `ps -p <pid> -o cmd` 核对命令行，只允许杀
  命令行中含本仓库 config 名（point2rbox_v2）的进程；批量 pkill 一律禁止

### 泄漏排查结论（修正）
- 隔离实验（固定合成 batch，300 步）lived_ops 恒定 41-52 → 模型/优化器路径无泄漏
- 前几次"渐进变慢+grad 卡顿"的主因高度疑似**多个残留训练进程同卡竞争**
  （最多时 2 个 jittor 训练 + torch baseline 同挤 GPU 0）——每次重启的旧世代
  没有被完全清理（杀父进程后 jittor 子进程孤儿化并继续持锁/持显存）
- 当前：单实例训练（18:41 起，带 lived_ops 打点），等 iter 500 数据最终确认

---

## 2026-07-26（Day 1，追加：性能问题全部结案）

### 结案报告（三个真问题 + 一个乌龙）
1. **diag3d python 循环建图**（真）：O(N) 图节点 → grad 分钟级。已修（批量闭式）
2. **逐实例 voronoi/L_target 循环**（真）：同类。已修（分块批量/argsort 分桶）
3. **_group_mean one-hot 矩阵乘**（真）：O(G·N·C)，高 num_pos 批次 10-20s/iter。
   已修（scatter-add，最坏批次实测 1.0s/iter 与 torch 持平）
4. **"周期性静默窗口"**（乌龙）：nohup 下 python stdout 块缓冲让日志成批冲刷，
   基于文件 mtime 的看门狗误报。**日志内嵌时间戳证明 20:00 的 run 全程无 >90s 间隔**。
   看门狗已改按日志内时间戳判停
- 附带排除项：图节点泄漏（liveness 打点平稳）、shape 重编译（无新 .so）、
  数据内容驱动（慢区批次实例数平常）
- 有效情报：jittor Dataset shuffle 跨运行确定性；数据 max 实例 648/patch

### 当前
- 训练（20:00 起，commit ba5b766）全速稳定，~2.5-2.8fps，ETA 明晨 ~4-5 点
- PyTorch baseline epoch 9/12（LR 已进入首个衰减档），ETA ~23:40

## Day 1 晚班（20:30-22:15）——"趁训练写代码"

### 完成
- **第 2 层数据集 loaders**（a264be3）：mm_datasets.py = DOTAv15/DOTAv2/STAR/RSAR/OCDPCB
  （实读 ref 归类：dior/fair 实为 XML 系，ocdpcb 实为 DOTA-txt；XML/COCO 系登记
  port_scope.md）；基类支持子类 CLASSES/IMG_SUFFIX + RSAR 扩展名回退；冒烟 8 项全过
- **stage-2 RotatedFCOSHead 全量 parity**（0a55c59）：golden=官方 using-pseudo 配置缩通道，
  loss 三项 rel<1e-3、forward 逐层<1e-3、cls+ctr 梯度 CPU 1e-4、总梯度 5e-3。
  过程中挖出并修掉三个真问题：
  1. diff_iou_rotated 鞋带公式在原始图像坐标下被 jittor FMA 融合放大大数消去
     （退化框假面积 → loss_bbox 差 16%）→ 改中心化坐标（面积平移不变）
  2. jittor nn.GroupNorm 一遍式方差 E[x²]−E[x]² → 底座 'GN' 换 GroupNorm2Pass
  3. CUDA torch dump golden 默认 TF32 → golden 自带 4e-4 污染 → dump 统一关
- **README 重写**（2db58e0）：v2 主叙事 + 结果表（TBD 等 ckpt）+ parity 表 + 上游折叠
- **issue #3 结案**（44d4f65）：B 的 bisect 指控 ba5b766 梯度回归 → 四步证据链证明是
  cudnn Find 计时选算法 + A100 TF32 候选的跨进程漂移假信号；scatter 梯度与 one-hot
  逐位一致；训练无恙不重启。复现脚本 tools/debug/。教训入 porting_notes：
  **GPU 数值 parity 不可用于 bisect，逐元素断言只在 CPU 做**
- HeadParity/FCOSHeadParity 重构为 GPU 松验(5e-2) + CPU 紧验(1e-4)，
  「独占 GPU 后复核容差」这一遗留项随之销案

### 当前
- 训练 iter 8350/epoch 1（21:42 时点），我的 parity 测试与其共卡一小时导致 fps
  2.83→2.18，测试已停、待回升；X baseline epoch 9/12 无恙
- 注意：训练进程载入的是 ba5b766 代码（GN 一遍式、旧 diff_iou）——语义无错不重启，
  stage-2 起自然用新代码

## Day 2 凌晨（00:35-00:50）——实验 X 收官

- **X（PyTorch baseline，同机同数据同 config）12ep 完成，val mAP50 = 54.50**
  （dota/mAP 0.5447；per-class 表在 expX log 00:43:31 处；强类 plane .881/ship .734、
  弱类 bridge .191/soccer .132 与论文端到端形态一致）
- **Jittor 端到端验收线就此确立：val mAP50 ∈ [53.5, 55.5]**（±1.0，同口径 val 评测）
- X ckpt (epoch_12.pth) 已投官方 pseudo-generator（pid 632205，GPU0 共卡，
  产物 data/split_ss_dota/point2rbox_v2_pseudo_labels.bbox.json）——
  作为 M7 伪标签 C4b json diff 的对照物 + 若 Jittor ckpt 有闪失时 stage-2 的备胎输入
- Jittor v2 epoch 2 完成时 fps 2.97，ETA 明晨 ~10:30
- pseudo-generator（X ckpt）完成：245,953 框 / 12,800 图，41.8MB json；
  已用我方 P2RV2DOTADataset._load_json 完整加载验证（字段/类别/尺寸全对）——
  stage-2 输入路径经真实产物打通，明天 C4b diff 的对照物就绪
