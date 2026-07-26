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
- M1 收尾：`tools/derive_scope.py` 机械化闭包 → `docs/port_scope.md` → push
- M2 开工：C1 梯度裁剪 / C2 组合 LR 调度（B 依赖，优先）
- L0 config parity 测试骨架

### 当前数字
- parity：尚未开始（M2 起有 L0/L1 数字）
- mAP：无
