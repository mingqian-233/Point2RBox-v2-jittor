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
