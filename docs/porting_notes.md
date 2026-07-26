# torch → Jittor 移植实战笔记（PLAN §8 的实测版）

> 全部条目都在本仓库实际踩过/验证过，L1/L2 测试锁定。给后来者（含 Agent B）参考。

## 🔴 会静默出错的（最危险，症状不指向根因）

| # | 问题 | 症状 | 对策 |
|---|---|---|---|
| 1 | **jittor 1.3.8.5 + numpy≥2** | 凡 `jt.array`（numpy 数据）喂进计算图的算子输出未初始化内存；roundtrip 与无输入算子（rand/ones）正常 | 钉死 numpy==1.26.4；smoke 测试有防回归断言 |
| 2 | **`jt.linalg.{det,inv,eigh,svd}` 是 numpy_code 实现** | GPU 上要 cupy（未装则 ModuleNotFoundError，且因惰性求值报错位置在千里之外的第一次 sync 处） | 2×2 全部用闭式解（`ops/linalg2x2.py`、`det_2x2`） |
| 3 | **in-place 赋值的自动微分不可靠** | 前向对、梯度 0 | loss 路径一律 out-of-place（jt.where/concat 重建）；L2 必须比梯度 |
| 4 | **`x.max()`/`.sum()` 等 reduce 返回 shape [1] 而非标量** | `jt.stack([a.max(), b.max()])` 得 (2,1)，下游 shape 错乱或 concat 报错 | 用 `jt.concat` 拼 reduce 结果 |
| 5 | **GPU 的 `jt.matmul` 不广播 batch 维**（cublas_batched_matmul） | CPU 通过、GPU 报 Wrong inputs | batch 维不一致时显式 `expand`（见 solve_2x2） |
| 6 | **conda 26.x `env config vars` 导出成大写** | jittor 读小写 `cc_path`，静默不生效 | 用 `etc/conda/activate.d/*.sh` |

## 🟡 语义差异（有对应物但行为不同）

| torch/mm 系 | jittor/JDet | 差异 |
|---|---|---|
| `torch.meshgrid(x, y, indexing='xy')` | 无 indexing 参数 | 手工构造：`X = x[None,:].expand((ny,nx))` |
| mmengine `LinearParamScheduler` | — | **分母是 end-begin-1**（≠torch LinearLR），第 W-1 iter 到顶 |
| mmcv `RoIAlignRotated(clockwise=True)` | JDet 原生方向 | **JDet 原生 = mmcv clockwise=True**（kernel 旋转矩阵互为转置）；aligned=-0.5 偏移需另加 |
| mmcv `nms_rotated` keep 按分数降序 | jdet 返回原始顺序 | head 里先 argsort 再 NMS |
| `torch.unique(return_inverse/counts)` | jt.unique 语义不保证 | detach 量一律走 numpy |
| `Tensor.index_reduce_('amin'/'mean')` | 无 | amin 于 unique 分组=组内代表（scatter）；mean 带梯度 → one-hot 矩阵乘 |
| `F.smooth_l1_loss(beta=)` | jt.nn.smooth_l1_loss 无 beta | 手写 `_smooth_l1` |
| torchvision `resized_crop`（antialias） | jt.nn.interpolate 无 antialias | 已记录差异（增广路径） |
| `nn.GroupNorm(requires_grad=)` | 无此参数 | config 适配层剥离该键 |
| mmengine cfg 的 `axis=` 遗留 | jt.concat 只认 `dim=` | 底座 coder.py 已修 |
| 底座 `RotatedResize` | mmdet Resize | 底座经 poly 往返会重规范角度（点框 0→-π/2）；自写 MMRotateResize |
| 底座 `RotatedRandomFlip` | mmdet RandomFlip | 底座 x'=W-x-1 且角度 π-a；mmrotate 是 W-x 与 -a；自写 MMRotateRandomFlip |
| 底座 `WhollyWoodDOTADataset` point_dummy=0.1 | 官方 ConvertWeakSupervision 默认 1 | v2 数据集显式用 1 |

## 🟢 直接可用（已验证）

`jt.concat/stack/split/gather/flip/clamp/where/arange/linspace/permute` /
`nn.grid_sample(含 reflection)` / `nn.affine_grid` / `nn.interpolate(bilinear, align_corners)` /
`nn.GroupNorm/PixelShuffle/ConvTranspose2d/MaxPool2d` / `jt.optim.AdamW(+clip_grad_norm 全局 L2)`。

## 数值/梯度专题

- **2×2 eigh 闭式解**（`ops/linalg2x2.py`）：`λ± = mean ± sqrt(clamp(delta²+b², 1e-24))`；
  EPS 取 1e-24（1e-12 会污染面积 1e-6 的框）；b 对称化读取让梯度在 [0,1]/[1,0] 对称分摊
  （对齐 torch.linalg.eigh 的梯度约定）；`|delta|/disc ≤ 1` ⇒ 特征值梯度天然有界，w==h 不炸。
- **简并（特征值重复）时**特征基与逐特征值子梯度不唯一：与 torch 的差异只在 a/c 间分配，
  trace 不变量一致。parity 测试对简并用例只比基无关量。
- **PSC decode ±π/2 端点**：atan2(∓0,-1) 符号翻转给出 +π/2 vs -π/2，le90 周期 π 下等价，
  按模 π 角距离比较。

## 工程

- Jittor 首跑 JIT 编译数分钟属正常；变长 shape 反复触发重编译（R5）——M5 实测后决定 bucket。
- 惰性求值：报错位置 ≠ 出错算子位置，二分法用 `.sync()`/`jt.flags.lazy_execution=0` 定位。
- 权重转换：torch `state_dict` → `{k: numpy}` pickle → `model.load_parameters`（TED 已验证）。
