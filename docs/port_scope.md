# 移植范围（port scope）——本仓库所有后续工作的边界

> 依据：PLAN §1 铁律三（按**移植成本**分层，不按"这次用不用得到"）。
> 机械化推导：`tools/derive_scope.py`（在 p2r-torch 环境跑，结果存
> `docs/port_scope_generated.md`，2026-07-26 生成，3 个官方 config 的并集）。
> **清单外的文件不移植**；中途发现确有间接依赖 → 在「中途追加」一节登记依赖路径后再动手。

## 第 1 层 — 核心实现（必须移植，主要工作量）

机械闭包（`port_scope_generated.md` §B）共 22 个文件，去掉 registry/`__init__` 后的实体：

| PyTorch 源（/root/ref/Point2RBox-v3/） | JDet 目标 | 说明 |
|---|---|---|
| `mmrotate/models/detectors/point2rbox_v2.py` | `python/jdet/models/networks/point2rbox_v2.py` | 18 KB |
| `mmrotate/models/dense_heads/point2rbox_v2_head.py` | `python/jdet/models/roi_heads/point2rbox_v2_head.py` | 43 KB，按方法块拆 commit |
| `mmrotate/models/dense_heads/rotated_fcos_head.py` | 扩展底座 `roi_heads/fcos_head.py` → `RotatedFCOSHead` | stage-2 用 |
| `mmrotate/models/losses/point2rbox_v2_loss.py` | `python/jdet/models/losses/point2rbox_v2_loss.py` | 4 个 loss 逐个移植；**以 v3 版为源**（含 v3 扩展参数，默认值=v2 行为，与 B 有约定） |
| `mmrotate/models/losses/gaussian_dist_loss.py` | 底座已有 → M2.5 核对 | |
| `mmrotate/models/losses/rotated_iou_loss.py` | JDet losses | stage-2 的 RotatedIoULoss |
| `mmrotate/models/losses/utils.py` | 随 loss 移植（闭包追加项） | shape metrics 工具，v3 的 mask 过滤用 |
| `mmrotate/models/losses/vis.py` | 随 loss 移植（闭包追加项，debug 可视化） | 若仅 debug 用可精简，需在此登记 |
| `mmrotate/models/task_modules/coders/angle_coder.py`（PSCCoder） | 底座已有（h2rbox_v2p 在用）→ M2.5 核对 | |
| `mmrotate/models/task_modules/coders/distance_angle_point_coder.py` | JDet coders | stage-2 用 |
| `mmrotate/datasets/transforms/transforms.py`（ConvertBoxType / ConvertWeakSupervision） | 底座 `data/transforms.py` 已有部分 → M2.5 核对语义 | |
| `mmrotate/evaluation/metrics/dota_metric.py`（DOTAMetric） | `python/jdet/data/dota.py` 的评测/输出路径 | C4 的 `format_only` json 输出必须与 mmrotate 逐字段一致 |
| `mmrotate/structures/bbox/transforms.py` | JDet `ops/bbox_transforms` 对应能力 | 闭包追加项 |
| `mmrotate/visualization/{local_visualizer,palette}.py` | JDet 可视化（低优先级，测试不依赖） | |
| `third_parties/ted/ted.py` | **与 B 协调**：TED 边缘检测被 detector import；B 的消息说他先做 TED | 见「与 B 的接口」 |
| `configs/point2rbox_v2/{point2rbox_v2-1x-dota,point2rbox_v2-pseudo-generator-dota,rotated-fcos-1x-dota-using-pseudo}.py` | `configs/point2rbox_v2/` | 逐行对照官方（PLAN §6） |
| `configs/_base_/{schedules/schedule_1x.py, default_runtime.py}` | 并入 JDet config（JDet 无 _base_ 继承时展平写） | |

**框架侧 type**（`port_scope_generated.md` §A 标"框架"的 30 项）不移植文件，
对应到 JDet core 能力：AdamW/LinearLR/MultiStepLR/clip_grad/EpochBasedTrainLoop/
SetEpochInfoHook 等 → M2 的 C1–C4；mmdet.ResNet/FPN/FocalLoss 等 → 底座已有实现，M2.5 核对。
`nms_rotated`（§A 唯一未解析项）是 mmcv.ops 算子，运行时经 `nms=dict(type='nms_rotated')`
字符串分发 → 对应底座 `python/jdet/ops/nms_rotated.py`，M2.5 做 L1 parity。

## 第 2 层 — 数据集定义 / 配置 / 注册项（全部包含，成本分钟级）

- 数据集定义（`mmrotate/datasets/`，10 个 + transforms/）：`dota.py`（含 v1.0/1.5/2.0）、
  `dior.py`、`hrsc.py`、`fair.py`、`rsar.py`、`star.py`、`sku110k.py`、`sardet100k.py`、
  `diatom.py`、`ocdpcb.py` → JDet 风格 loader：
  - 底座已有 `dota.py`/`fair.py`/`ssdd_plus.py`/`coco.py`/`custom.py`/`whollywood_dota.py` → 复用
  - DOTA 标注格式系（dior/star/rsar/…）→ 从 DOTA loader 派生换 METAINFO，每个约十几分钟
  - 异构格式（hrsc=XML、sku110k=csv、sardet100k=COCO、diatom/ocdpcb）→ 排交付后期，
    必须能 import 能实例化；来不及则在此文件显式登记未完成项

### 数据集移植清单（2026-07-26 实做核查后更新）

实读 ref 源码后格式归属与上面的预估有出入（dior/fair 实为 XML、ocdpcb 实为 DOTA-txt）：

| 数据集 | ref 实际格式 | 状态 |
|---|---|---|
| DOTA v1.0 | DOTA-txt | ✅ `p2rv2_dota.py::P2RV2DOTADataset`（主线，全 parity） |
| DOTA v1.5 / v2.0 | DOTA-txt | ✅ `mm_datasets.py`（16/18 类，METAINFO 逐字对齐） |
| STAR | DOTA-txt（48 类，跳未知类） | ✅ `mm_datasets.py` |
| RSAR | DOTA-txt（6 类，图像扩展名不统一） | ✅ `mm_datasets.py`（基类扩展名回退） |
| OCD-PCB | DOTA-txt（41 类，.png） | ✅ `mm_datasets.py` |
| DIOR | **XML**（两分支均读 `Annotations/*.xml`） | ✅ `dior.py`（robndbox 8 点；支持 train+val id 列表） |
| FAIR1M | XML 原图，经官方预处理转 DOTA-txt | ✅ 复用底座 `fair.py` + `devkits/fair_to_dota.py` 既有链路 |
| HRSC2016 | XML | ✅ `hrsc.py`（mbox cx/cy/w/h/angle，默认单类语义） |
| DIATOM | XML（单类） | ✅ `diatom.py`（hbb → rbox(angle=0)） |
| SKU110K | 图像目录 + json 分支（单类） | ✅ `sku110k.py`（rbbox 优先，兼容 hbb） |
| SARDet-100k | COCO（BaseDetDataset） | ✅ `coco_rbox.py`（COCO bbox/8 点 segmentation → rbox） |

冒烟测试：`tests/smoke/test_mm_datasets.py`（注册/实例化/标签映射/difficulty 过滤/
未知类跳过/空图过滤/RSAR 扩展名回退/基类回归，全绿）。
异构 loaders 复用 Agent B 已做过的实现并在 A 仓库重新验收；测试见
`tests/smoke/test_heterogeneous_datasets.py`（注册、ref 类别表、合成 XML/JSON 解析）。
- 数据集 config（`configs/_base_/datasets/`）：**全部 19 个**
- 注册链：`datasets/__init__.py`、`structures/bbox/*`、`evaluation/**/__init__.py`

## 第 3 层 — 不移植（黑名单）

- 🚫 ReDet 等变网络：`backbones/re_resnet.py`、`necks/re_fpn.py`、`models/utils/{enn,orconv,ripool}.py`（依赖 e2cnn，最大的坑）
- 🚫 `models/roi_heads/**`（v2 与 stage-2 都是 single-stage anchor-free）
- 🚫 全部 assigners（闭包证实 head 只 build PSCCoder / DistanceAnglePointCoder）
- 🚫 其余 21 个 dense_heads、11 个 detectors、10 个 losses、9 个 coders
- 🚫 未移植方法的 config（`configs/{h2rbox,h2rbox_v2,point2rbox,whollywood,point2rbox_v3}/**`；
  数据集 config 除外——那是第 2 层）
- 🚫 `resources/`、`poster/`、`figures/`
- **反向规则**：JDet 底座里的无关代码（projects/、redet、yolo、s2anet…）**保留不动**

## 机械闭包 vs 计划清单的差异（`[plan-deviation]` 记录)

1. **计划多列的**：`task_modules/synthesis_generators/point2rbox_generator.py`——
   闭包证实只被 v1/whollywood/yolof 引用，**v2 不 import**（v2 的 copy-paste 合成在 head 内部实现）。
   处置：不移植、不核对；底座已有的文件保留不动。
2. **计划漏列、闭包补上的**：`losses/utils.py`、`losses/vis.py`、`structures/bbox/transforms.py`、
   `visualization/{local_visualizer,palette}.py`、`third_parties/ted/ted.py`、`registry.py`（注册链）。
3. `nms_rotated` 不经 registry（mmcv.ops 运行时分发），脚本标"未解析"属预期，已人工归入 ops parity。

## 与 B 的接口（涉及本清单的）

- `third_parties/ted/ted.py`：B 2026-07-26 12:19 消息称"我先做 TED"（v3 侧）。TED 同时被
  v2 detector import——**M4 移植 detector 前在 COORD 对齐**：复用 B 的实现还是我各自移植。
- `point2rbox_v2_loss.py` 以 v3 版为源（含 SAM 扩展参数），SAM predictor 依赖接口由 B 提供。

## 中途追加项（追加式，动手前登记）

<!-- 格式：日期 / 文件 / 依赖路径（谁 import 它）/ 层级 -->
