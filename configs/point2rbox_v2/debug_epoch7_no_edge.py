"""Resume ckpt_6 for one epoch with copy-paste on and EdgeLoss off."""

_base_ = './point2rbox_v2_1x_dota.py'

name = 'debug_epoch7_no_edge'
work_dir = 'work_dirs/debug_epoch7_no_edge'
resume_path = 'work_dirs/point2rbox_v2_1x_dota/checkpoints/ckpt_6.pkl'
eval_interval = 1
checkpoint_interval = 1

model = dict(bbox_head=dict(edge_loss_start_epoch=999))
