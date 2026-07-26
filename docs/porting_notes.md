# torch → Jittor 移植实战笔记（PLAN §8 的实测版）

> 全部条目都在本仓库实际踩过/验证过，L1/L2 测试锁定。给后来者（含 Agent B）参考。

## 🔴 会静默出错的（最危险，症状不指向根因）

| # | 问题 | 症状 | 对策 |
|---|---|---|---|
| 0a | **`.stop_grad()` 是就地打标记（≠torch detach）** | 把它当 detach 用 → 原变量本体被标无梯度，前向全对、相关 loss 梯度静默为 0 | detach 语义一律用 `.detach()`（返回新变量）；`.stop_grad()` 只用于"这个变量从此不需要梯度"（如冻结参数） |
| 0b | **任何 `for x in tensor` 式 python 循环建图 = O(N) 图节点** | 密集批次 grad/sync 阶段 100% CPU 停滞分钟级、位置随 shuffle 漂移、无报错。实锤案例：底座 `diag3d`（循环 jt.diag，gwd_loss 每 iter ×3）、逐实例 voronoi、逐实例 nonzero | 一律批量闭式改写（stack/where/searchsorted 分桶）；detach 路径转 numpy；用 `jt.liveness_info()` 打点确认 lived_ops 有界 |
| 0c | **多进程 dataloader 环形缓冲死锁** | worker 卡 buffer.send、主进程卡 idqueue.pop（py-spy 可证） | num_workers=0（加载占比小时性价比最高） |
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
- **diff_iou_rotated 的鞋带公式必须在中心化坐标上算**（2026-07-26，stage-2 parity 揪出）：
  mmcv 在原始图像坐标（~1e2-1e3）上算，x_i·y_j 项达 1e4-1e6，微小/退化交集依赖正负大项
  相消；jittor 编译器把 `a*d - b*c` 融合成 FMA 后两项舍入不再互为相反数，残差 ~1e-2 px²
  变成假面积（golden 上 loss_bbox 差 16%，torch eager 恰好精确抵消得 0）。修法：排序仍在
  归一化坐标做（不变），面积改在「减均值后再乘 mask 归零填充槽」的坐标上算——闭合多边形
  鞋带公式平移不变，且填充槽零贡献保持。torch 侧原始坐标的噪声底 ~1e-3 仍留在 golden 的
  loss_bbox 梯度里，故 FCOS head 梯度紧验走 cls+ctr 支路（1e-4），总梯度 5e-3。
- **jittor nn.GroupNorm 是一遍式方差** `E[x²]−E[x]²`（灾难性消去风险），torch 是两遍式
  `E[(x−mean)²]`。底座 BRICKS 'GN' 已换成 `GroupNorm2Pass`（modules.py，签名/参数名兼容）。
- **golden 若在 CUDA torch 上 dump，必须关 TF32**：torch 2.x 默认 `cudnn.allow_tf32=True`，
  卷积只有 10 位尾数，前向自带 ~4e-4 相对误差，会整体污染 golden（曾被误判为 GN/实现差异，
  实际逐层 CPU 对比 conv 输出 rel 1e-7）。dump 脚本统一加
  `torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False`。
- **torch `ReLU(inplace=True)` 会篡改中间量取证**：dump 中间激活时先 `.clone()` 再过激活，
  否则 pre-act 张量被原地覆盖，逐层对比出现「norm 层 rel=1.0 而 act 后 rel=1e-7」的假象。
- **GPU 数值 parity 不可用于 bisect / 逐元素断言**（issue #3 结案，2026-07-26 22:00）：
  jittor conv 前向/反向用 `cudnnFindConvolution*AlgorithmEx` **真实计时**选算法并做
  **进程内 cache**；cudnn8 在 A100 上 `CUDNN_DEFAULT_MATH` 允许 TF32 tensor-core 算法
  参与候选。选中 TF32 的进程 GPU-vs-golden 前向 ~4e-4、梯度 rel_l2 ~4e-2（逐元素违约
  ~22%）；选中 FMA 的进程 ~1e-4。结果「进程内确定、跨进程漂移」，且**换 commit 改图形状
  会扰动 benchmark 计时** → bisect 会把算法选择的翻转误报成某个 commit 的数值回归
  （B 对 ba5b766 scatter-add 的 bisect 即此假信号；scatter vs one-hot 实测梯度逐位一致，
  复现脚本 tools/debug/）。此前文档写的「共卡 cudnn 漂移、偶发 ~4%」是同一现象的
  不完整解释。对策：梯度验收 = GPU 松验（rel_l2<5e-2，兜断链级 bug）+ CPU 紧验
  （rel_l2<1e-4）；训练不受影响（TF32 卷积训练是业界默认，torch 侧同样如此）。

## 工程

- Jittor 首跑 JIT 编译数分钟属正常；变长 shape 反复触发重编译（R5）——M5 实测后决定 bucket。
- 惰性求值：报错位置 ≠ 出错算子位置，二分法用 `.sync()`/`jt.flags.lazy_execution=0` 定位。
- 权重转换：torch `state_dict` → `{k: numpy}` pickle → `model.load_parameters`（TED 已验证）。
