"""One-epoch diagnostic resumed at ckpt_6 with edge on, copy-paste off."""

_base_ = './point2rbox_v2_1x_dota.py'

name = 'debug_epoch7_no_copy_paste'
work_dir = 'work_dirs/debug_epoch7_no_copy_paste'
resume_path = 'work_dirs/point2rbox_v2_1x_dota/checkpoints/ckpt_6.pkl'
max_epoch = 7
eval_interval = 1
checkpoint_interval = 1

model = dict(copy_paste_start_epoch=999)
