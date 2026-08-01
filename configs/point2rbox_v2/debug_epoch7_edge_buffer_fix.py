"""Replay epoch 7 from ckpt6 with frozen/reconstructed EdgeLoss buffers."""

_base_ = './point2rbox_v2_1x_dota.py'

name = 'debug_epoch7_edge_buffer_fix'
work_dir = 'work_dirs/debug_epoch7_edge_buffer_fix'
resume_path = 'work_dirs/point2rbox_v2_1x_dota/checkpoints/ckpt_6.pkl'
eval_interval = 1
checkpoint_interval = 1
