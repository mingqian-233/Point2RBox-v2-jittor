# debug 复现脚本（issue #3 结案证据）

结论与机制见 docs/porting_notes.md「GPU 数值 parity 不可用于 bisect」条目。

- `repro_scatter_vs_onehot_grad.py` — _group_mean 的 scatter-add vs one-hot：
  CPU/GPU 前向 ~1e-7、梯度逐位一致（maxdiff=0）
- `repro_head_gpu_vs_cpu.py` — v2 head 整头同代码 GPU vs CPU 三角验证（viol=0）
- `repro_gpu_golden_process_drift.py` — GPU-vs-golden 跨进程漂移 + GN 实现对照
  （同进程内换 GN 实现结果几乎不变 → 与算子无关，是 cudnn 算法选择）

运行：`CUDA_VISIBLE_DEVICES=0 cc_path=/usr/bin/g++-10 python tools/debug/<script>.py`
