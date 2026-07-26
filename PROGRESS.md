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
