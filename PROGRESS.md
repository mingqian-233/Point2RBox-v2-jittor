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
