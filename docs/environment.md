# 环境记录（实测生效的版本组合 + 踩坑）

> 一键搭建：`bash scripts/setup_env.sh`。本文记录**为什么**是这些版本——改动前先读踩坑记录。

## 机器

| 项 | 值 |
|---|---|
| OS | Ubuntu 24.04（glibc **2.39**，这是几个坑的根源） |
| CPU / RAM | 96 核 AMD EPYC 7V13 / 866 GB |
| GPU | 4 × A100 80GB PCIe（本仓库只用 `CUDA_VISIBLE_DEVICES=0`） |
| 驱动 | 580.105.08（CUDA 13.0 driver API） |
| 编译器 | 系统 g++ 13.3.0；**jittor 实际使用 g++-10.5**（见坑 #2） |

## env 1：`p2r-jittor`（移植/训练）

| 包 | 版本 | 说明 |
|---|---|---|
| python | 3.10.20 | 对齐参考仓库 whollywood-jittor |
| jittor | **1.3.8.5** | 对齐参考仓库 |
| numpy | **1.26.4** | **绝不能是 2.x**，见坑 #1 |
| CUDA（jittor 用） | jtcuda cuda11.2_cudnn8（jittor 自动下载） | A100 sm_80，驱动 580 向后兼容，实测可用 |
| 编译器 | `cc_path=/usr/bin/g++-10`（已写进 conda env config vars） | 见坑 #2 |

验收（均通过，2026-07-26）：
- `python -m jittor.test.test_example` exit 0，loss 收敛到 1e-3
- GPU 1000×1000 matmul 与 numpy 最大误差 2.1e-4（float32 正常范围）
- cudnn conv 前向可用

## env 2：`p2r-torch`（golden reference，严格对齐 `Point2RBox-v3/environment.md`）

| 包 | 版本 |
|---|---|
| python | 3.12.13 |
| torch / torchvision | 2.2.0+cu121 / 0.17.0+cu121 |
| mmengine / mmcv / mmdet | 0.10.7 / 2.2.0 / 3.3.0 |
| mmrotate | 1.0.0rc1（`pip install -e /root/ref/Point2RBox-v3`） |
| numpy | **1.26.4**（mm 系列装完后重新钉回） |
| 其他 | scipy 1.17.1、opencv-python 4.11.0.86、timm、segment-anything、mobile_sam（源码装）、pandas（见坑 #5） |

验收（2026-07-26）：版本自检输出与上游 environment.md 预期完全一致；
`python tools/train.py configs/point2rbox_v2/point2rbox_v2-1x-dota.py` 能完成
registry 注册、config 解析、模型构建，最终停在
`ValueError: There is no txt file in data/split_ss_dota/trainval/annfiles/`（数据未就绪，符合预期）。

## 踩坑记录

### 坑 #1（最危险）：jittor 1.3.8.5 + numpy 2.x = 静默数值错误

pip 装 jittor 会自动带上 numpy 2.2.6。此组合下 **jittor 不报任何错误**，但凡是从
`jt.array`（numpy 数据）喂进计算图的算子，输出全是未初始化内存的垃圾值：

```python
a = jt.float32([1,2,3])
(a+a).numpy()   # → [1.05e-17, 0.0, 0.0]  而不是 [2,4,6]，无任何报错！
```

诡异之处：`jt.array(x).numpy()` 往返是对的、无输入的算子（`jt.rand`/`jt.ones`）也是对的，
只有"吃 numpy 数据的计算"错——极难从训练现象反推。表现为 `test_example` loss=NaN。
**修复：`pip install numpy==1.26.4`（清掉 `~/.cache/jittor` 重编译后生效）。**
每次 pip 装新包后务必 `pip show numpy` 复查有没有被顶回 2.x。

### 坑 #2：g++ 版本链（glibc 2.39 × nvcc 11.2）

jittor 自带 jtcuda 的 nvcc 是 11.2，在 Ubuntu 24.04 上：

| 编译器 | 结果 |
|---|---|
| g++-13（系统默认） | ❌ nvcc 前端不认 `__builtin_dynamic_object_size`（glibc fortify 头文件在 GCC≥12 时启用） |
| g++-11（计划里的预案） | ❌ nvcc 前端不认带参数的 `__attribute__((__malloc__(x,y)))`（glibc 2.34+ 在 GCC≥11 时启用） |
| **g++-10** | ✅ 全部编译通过 |

修复：`apt-get install g++-10`，并在 env 的 `etc/conda/activate.d/jittor_cc.sh` 里
`export cc_path=/usr/bin/g++-10`（jittor 读 `cc_path` 环境变量选编译器，nvcc 的 `-ccbin` 跟随它）。
⚠️ **不能用 `conda env config vars set`**：conda 26.x 会把变量名导出成大写 `CC_PATH`，
jittor 只认小写，静默不生效（B 实测踩坑，2026-07-26 12:33）。
注意：数值垃圾问题（坑 #1）与编译器无关——g++-10/13 下都复现，钉 numpy 才是修复。

### 坑 #3：conda 26.x 首次使用需接受 ToS

`conda create` 会报 `CondaToSNonInteractiveError`。修复：
`conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`（r 频道同理）。

### 坑 #4：mmdet 3.3.0 的 mmcv 版本上界

`import mmdet` 断言 `mmcv<2.2.0`。上游 environment.md FAQ 官方推荐改法：把
`mmdet/__init__.py` 的 `mmcv_maximum_version` 从 `'2.2.0'` 改成 `'2.3.0'`。已改。

### 坑 #5：ref 仓库的 3 处 IDE 误加 import

`point2rbox_v2_loss.py` 有 `from pandas import Timestamp` 等 3 处无用 import
（计划 §8 已预告）。ref 仓库只读不能删 → 在 p2r-torch 里装了 pandas 让 import 通过
（sympy、click 已是其他包的传递依赖）。**移植到本仓库时这些 import 直接删掉。**

### 坑 #6：跑 ref 仓库需要 PYTHONPATH

`point2rbox_v2` detector 依赖仓库根目录的 `third_parties/`（TED 边缘检测），
editable install 只暴露 `mmrotate` 包。跑 ref 侧任何脚本都要：

```bash
cd /root/ref/Point2RBox-v3 && PYTHONPATH=/root/ref/Point2RBox-v3 python tools/train.py ...
```

## 与计划（PLAN-AGENT-A.md）的偏差

- `[plan-deviation]` 计划 M0 预案是 g++-11，实测 g++-11 也不行，最终用 g++-10（坑 #2）。
- `[plan-deviation]` 计划未预料 numpy 2.x 会静默破坏 jittor 数值（坑 #1），两个 env 都钉 1.26.4。
- `[plan-deviation]` p2r-torch 额外装了 pandas（坑 #5，ref 只读所致）。
- jittor 未动用回退（1.3.8.5 原版可用，未升级版本、未用系统 CUDA、未降级 CPU）。
