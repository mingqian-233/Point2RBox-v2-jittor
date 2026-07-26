# 项目状态一览（快照：2026-07-26 19:20，Day 1 结束前）

> 给后来接手者的单页地图。历史细节看 `PROGRESS.md`（追加式日报），
> 本文只保留"现在是什么状态、东西在哪、接下来做什么"。
> 计划原文：`/root/work/A/PLAN-AGENT-A.md`；与 Agent B 的通信：`/root/work/COORD.md`。

## 一句话

把 Point2RBox-v2 从 PyTorch/mmrotate 移植到 Jittor/JDet 并在 DOTA-v1.0 复现精度。
**Day 1 完成了 M0→M4 全部 + M7 全部代码件；两条 12-epoch 训练在跑；
所有移植均有数值 parity 测试背书（49 commits，45+ tests 全绿）。**

## 里程碑状态

| 里程碑 | 状态 | 关键证据 |
|---|---|---|
| M0 环境（p2r-jittor / p2r-torch） | ✅ | `scripts/setup_env.sh`、`docs/environment.md`（6 坑） |
| M1 骨架 + 移植边界 | ✅ | `docs/port_scope.md`（机械化闭包推导） |
| M2 core（clip_grad/LR/epoch注入）+ M2.5 复用件核对 | ✅ | LR 与 mmengine 1440 点逐点相等 |
| M3 四个 loss + tag **core-stable** | ✅ | 前向<1e-4 / 梯度<1e-3，B 已 merge |
| M4 head+detector+数据集+config | ✅ | **固定权重下 6 项 loss_dict 与 torch rel<1e-3** |
| L0/L1/L2/L3 测试体系 | ✅ | L3=真实训练权重整模型前向 rel L2 ≤1.2e-3 |
| M6 端到端训练（实验 #1） | 🔄 在跑 | 见下"在跑的东西" |
| 实验 X（PyTorch baseline 对照组） | 🔄 epoch 8/12 | ETA 今晚 ~23:30 |
| M7 两阶段（代码件） | ✅ 代码就绪 | 伪标签工具链 + RotatedFCOSHead + RotatedIoULoss 全 parity |
| M7 两阶段（执行） | ⬜ 等 #1 ckpt | 明天 |
| M8 交付（README/HF/提交包） | ⬜ | 等训练结果 |

## 在跑的东西（重启后自查这里）

| 任务 | 命令/日志 | 状态 |
|---|---|---|
| 实验 X：PyTorch baseline | `/root/work/A/torch_baseline/logs/expX_*.log`，work_dir 同目录 | epoch 8/12，loss 稳降，edge loss epoch7 已激活 |
| Jittor v2 正式 12ep | 仓库根 `logs/train_v2_official.log`，`work_dirs/point2rbox_v2_1x_dota/` | 19:07 起（commit 0813ffe 含全部修复），ETA 明天中午 |
| 训练启动命令 | `CUDA_VISIBLE_DEVICES=0 cc_path=/usr/bin/g++-10 python tools/run_net.py --config-file=configs/point2rbox_v2/point2rbox_v2_1x_dota.py --task=train` | ⚠️ 必须从仓库根启动；启动后必须验证进程存活 |
| GPU 纪律 | A=GPU0，B=GPU1，GPU2/3 用户自用**禁碰** | kill 前必须 `ps -o cmd` 核对（见事故） |

## 必读的坑（按杀伤力排序，详见 docs/porting_notes.md / environment.md）

1. `.stop_grad()` 是**就地**打标记 ≠ torch detach → 用 `.detach()`（计划 §8 的建议本身是错的）
2. 任何 `for x in tensor` python 循环建图 = O(N) 图节点 → grad 卡死分钟级（实锤：底座 `diag3d`、逐实例 voronoi）
3. jittor 1.3.8.5 + numpy≥2 = 静默数值垃圾 → 钉死 numpy==1.26.4
4. `jt.linalg.*` GPU 要 cupy → 2×2 全用 `ops/linalg2x2.py` 闭式解
5. 底座 nms_rotated CUDA 宿主解引用 device 指针 → 已修（4c15ae7）
6. 底座 FocalLoss 是 1-based 标签 → v2/stage-2 用 `MMDetFocalLoss`
7. GPU matmul 不广播 batch 维（CPU 会）→ 显式 expand
8. jittor 多进程 dataloader 环形缓冲死锁 → num_workers=0（唯一 plan-deviation 的 config 项）
9. 编译器必须 g++-10（`cc_path`，activate.d 已配）；conda env vars 会被导出成大写

## 事故记录

- **18:44 误杀 Agent B 的 v3 训练**（清理孤儿进程时未核对命令行）。已在 COORD 报告致歉，
  B 已 resume。立规：kill 前必须核对 `ps -o cmd`，只杀含 `point2rbox_v2` 的进程，禁批量 pkill。
- 训练卡死三连（dataloader 死锁 / 逐实例循环巨图 / diag3d）已全部根因修复，
  泄漏假设被隔离实验+liveness 打点排除。

## 剩余工作（依赖序）

1. 【等 X ckpt，今晚】X 的 val mAP → 确立本机 baseline 验收线（±1.0 mAP50）；
   X ckpt 跑官方 pseudo-generator → M7 伪标签对照物 + C4b json diff
2. 【等 Jittor ckpt，明天】val/test 评测 → 打 tag `v2-stable` → 伪标签生成
   （`tools/generate_pseudo_labels.py`）→ stage-2 训练（config 已备）
3. stage-2 head 的 torch parity golden（开训前，半小时）
4. 铁律三第 2 层数据集 loaders（dior/star/rsar 分钟级；hrsc/sku110k 等异构中等，
   可延后但需在 port_scope.md 登记）
5. `--task=test` 的 merge_patches 验证 + DOTA 提交包
6. M8：README 重写 / HF 上传 / 20-iter 曲线对照补充
7. 独占 GPU 后复核 HeadParity feat 梯度容差（现 5e-2，共卡 cudnn 漂移）

## 需要用户做的两件事

- **CI**：token 缺 `workflow` scope，push workflows 会被拒 → 去 GitHub token 设置勾选（COORD 有说明）
- **DOTA test 提交**：官方评测服务器需账号，训练完我备好 zip 后由用户上传

## 文件地图

```
configs/point2rbox_v2/           # 三个 config（端到端/伪标签/stage-2），L0 逐值锚定
python/jdet/models/
  networks/point2rbox_v2.py      # detector（dual-stream/copy-paste/TED）
  roi_heads/point2rbox_v2_head.py# v2 head（43KB 源的移植）
  roi_heads/rotated_fcos_head.py # stage-2 head
  losses/point2rbox_v2_loss.py   # 四个 loss + MMDet{Focal,CE}Loss
  losses/rotated_iou_loss.py     # 可微旋转 IoU
python/jdet/ops/{linalg2x2,diff_iou_rotated}.py   # 自研算子
python/jdet/data/p2rv2_dota.py   # 直读 split_ss_dota + bbox.json
third_parties/ted/               # TED 边缘模型（自包含）
tools/                           # dump_golden_*.py / convert_*_ckpt.py / generate_pseudo_labels.py
tests/parity/                    # L0-L3 + golden（npz/json 全部入库）
docs/                            # environment/port_scope/config_parity/porting_notes/STATUS
```

## 协作状态（Agent B）

- B 已 merge 我的 `core-stable`；targets/bids 约定已对齐（COORD 13:31）
- 双向共享的坑均已通报；B 的 TED/SAM 与我的 third_parties 版并存不冲突
- B 在等我的 `v2-stable` tag 开他的 M5
