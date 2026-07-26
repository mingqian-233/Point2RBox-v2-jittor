#!/usr/bin/env bash
# setup_env.sh — 从零搭建 Point2RBox-v2-jittor 的两套 conda 环境
#   p2r-jittor : Jittor 1.3.8.5 移植/训练环境（Python 3.10）
#   p2r-torch  : PyTorch golden reference 环境（Python 3.12，对齐 Point2RBox-v3/environment.md）
#
# 实测环境：Ubuntu 24.04 (glibc 2.39) / A100 80GB / driver 580.105.08 (CUDA 13.0)
# 可重复执行（幂等性尽力而为：conda env 已存在时跳过创建）。
# 踩坑记录见 docs/environment.md —— 改动本脚本前先读它。
set -euo pipefail

MINICONDA_PREFIX=/opt/miniconda3
REF_DIR=/root/ref

# ---------- 1. 系统依赖 ----------
# g++-10 是硬要求：jittor 自带的 nvcc 11.2 不兼容 g++-11/13（见 docs/environment.md）
apt-get update
apt-get install -y build-essential g++ g++-10 libomp-dev libgl1 libglib2.0-0 \
    wget curl unzip zip git git-lfs

# ---------- 2. Miniconda ----------
if [ ! -x "$MINICONDA_PREFIX/bin/conda" ]; then
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$MINICONDA_PREFIX"
fi
"$MINICONDA_PREFIX/bin/conda" init bash
# 新版 conda 需要先接受 ToS 才能用默认 channel
"$MINICONDA_PREFIX/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
"$MINICONDA_PREFIX/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
source "$MINICONDA_PREFIX/etc/profile.d/conda.sh"

# ---------- 3. p2r-jittor ----------
if ! conda env list | grep -q '^p2r-jittor '; then
    conda create -n p2r-jittor python=3.10 -y
fi
conda activate p2r-jittor
pip install jittor==1.3.8.5
# ⚠️ 关键：jittor 1.3.8.5 + numpy>=2 会静默产出错误数值（无报错！），必须钉死 1.26.4
pip install numpy==1.26.4
# ⚠️ 关键：jittor 编译必须用 g++-10。注意不能用 `conda env config vars set`——
# conda 26.x 会把变量名导出成大写 CC_PATH，而 jittor 只认小写 cc_path。用 activate.d 脚本：
ENVDIR="$MINICONDA_PREFIX/envs/p2r-jittor"
mkdir -p "$ENVDIR/etc/conda/activate.d" "$ENVDIR/etc/conda/deactivate.d"
printf 'export cc_path=/usr/bin/g++-10\n' > "$ENVDIR/etc/conda/activate.d/jittor_cc.sh"
printf 'unset cc_path\n' > "$ENVDIR/etc/conda/deactivate.d/jittor_cc.sh"
conda deactivate && conda activate p2r-jittor
# 首次 import 会自动下载 jtcuda(cuda11.2_cudnn8) 并编译内核（数分钟）
python -m jittor_utils.install_cuda || true
python -m jittor.test.test_example
# 数值正确性冒烟（防止 numpy/编译器问题静默破坏数值）
python - <<'PY'
import jittor as jt, numpy as np
a = jt.float32([1, 2, 3])
assert (a + a).numpy().tolist() == [2., 4., 6.], 'elementwise broken!'
jt.flags.use_cuda = 1
x, y = jt.rand(100, 100), jt.rand(100, 100)
err = abs(x.matmul(y).numpy() - x.numpy() @ y.numpy()).max()
assert err < 1e-3, f'gpu matmul err={err}'
print('p2r-jittor OK')
PY
conda deactivate

# ---------- 4. p2r-torch（严格对齐 Point2RBox-v3/environment.md）----------
if ! conda env list | grep -q '^p2r-torch '; then
    conda create -n p2r-torch python=3.12 -y
fi
conda activate p2r-torch
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmengine==0.10.7
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
pip install mmdet==3.3.0
# mm 系列会把 numpy 顶到 2.x，装完必须钉回
pip install numpy==1.26.4
pip install scipy pillow shapely opencv-python pycocotools matplotlib timm segment-anything
# pandas：ref 仓库 point2rbox_v2_loss.py 有一处 IDE 误加的 `from pandas import Timestamp`，
# ref 只读不可删，只能装上让 import 通过
pip install pandas
[ -d "$REF_DIR/MobileSAM" ] || git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM.git "$REF_DIR/MobileSAM"
pip install -e "$REF_DIR/MobileSAM"
# mmdet 3.3.0 的 mmcv 上界写死 <2.2.0，实际 2.2.0 可用（上游 FAQ 官方推荐的改法）
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
    "$MINICONDA_PREFIX/envs/p2r-torch/lib/python3.12/site-packages/mmdet/__init__.py"
pip install -v -e "$REF_DIR/Point2RBox-v3"
python -c "
import torch, torchvision, mmcv, mmdet, mmengine, mmrotate, numpy
assert numpy.__version__ == '1.26.4', numpy.__version__
print('torch', torch.__version__, '| mmcv', mmcv.__version__, '| mmdet', mmdet.__version__,
      '| mmrotate', mmrotate.__version__, '| cuda', torch.cuda.is_available())
print('p2r-torch OK')
"
conda deactivate

echo "=== 全部完成。用法： ==="
echo "  conda activate p2r-jittor   # Jittor 移植/训练"
echo "  conda activate p2r-torch    # golden reference（跑 ref 仓库需 PYTHONPATH=$REF_DIR/Point2RBox-v3）"
