"""Dump one deterministic official PyTorch train step (losses + gradients)."""
import argparse
import math

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes

from dump_official_ss_predictions import CLASSES, load_ann


WATCH = (
    'backbone.layer2.0.conv1.weight',
    'backbone.layer3.0.conv1.weight',
    'backbone.layer4.0.conv1.weight',
    'neck.lateral_convs.0.conv.weight',
    'bbox_head.cls_convs.0.conv.weight',
    'bbox_head.reg_convs.0.conv.weight',
    'bbox_head.conv_cls.weight',
    'bbox_head.conv_reg.weight',
    'bbox_head.conv_angle.weight',
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--ann', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ss-mode', choices=('rot', 'flp', 'sca'), default='rot')
    p.add_argument('--ss-value', type=float, default=None,
                   help='Rotation in degrees or scale factor; defaults to 67.5/1.25.')
    p.add_argument('--epoch', type=int, default=1)
    p.add_argument('--fixed-copy-paste', action='store_true')
    p.add_argument('--dump-generated-cache', action='store_true')
    args = p.parse_args()

    init_default_scope('mmrotate')
    model = MODELS.build(Config.fromfile(args.config).model).cuda()
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.train()
    model.set_epoch(args.epoch)
    if args.ss_mode == 'rot':
        fixed = (67.5 if args.ss_value is None else args.ss_value) / 180.0
        model.ss_prob = [1.0, 0.0, 0.0]
        model.rotate_range = (fixed, fixed)
    elif args.ss_mode == 'flp':
        model.ss_prob = [0.0, 1.0, 0.0]
    else:
        fixed = 1.25 if args.ss_value is None else args.ss_value
        model.ss_prob = [0.0, 0.0, 1.0]
        model.scale_range = (fixed, fixed)

    bgr = cv2.imread(args.image)
    rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().cuda()
    mean = torch.tensor([123.675, 116.28, 103.53], device='cuda')[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375], device='cuda')[:, None, None]
    image = ((rgb - mean) / std)[None]
    boxes, labels = load_ann(args.ann)
    boxes[:, 2:4] = 1.0
    boxes[:, 4] = 0.0
    ds = DetDataSample(
        metainfo=dict(img_shape=(1024, 1024), ori_shape=(1024, 1024),
                      scale_factor=(1.0, 1.0)))
    ds.gt_instances = InstanceData(
        bboxes=RotatedBoxes(torch.from_numpy(boxes).cuda()),
        labels=torch.from_numpy(labels).cuda())
    if args.fixed_copy_paste:
        # A framework-independent RGBA patch.  Keeping this synthetic removes
        # random pattern generation from the copy-paste application test.
        yy, xx = np.mgrid[:20, :24].astype(np.float32)
        pattern = np.stack((
            (xx - 12.0) / 12.0,
            (yy - 10.0) / 10.0,
            (xx + yy - 22.0) / 22.0,
            np.full_like(xx, 0.75)), axis=0).astype(np.float32)
        patch_box = np.asarray([12.0, 10.0, 16.0, 8.0, 0.2], np.float32)
        model.copy_paste_cache = [[(
            torch.from_numpy(pattern).cuda(), patch_box.copy(), 4)]]
        np.random.seed(314159)

    # NumPy drives pattern geometry and paste offsets in the official code.
    np.random.seed(314159)
    consistency_inputs = {}
    def _capture_consistency(_module, inputs):
        (go, ao), (gt, at), sq, aug_type, aug_val = inputs
        consistency_inputs.update(
            con_go=go.detach().cpu().numpy(),
            con_ao=ao.detach().cpu().numpy(),
            con_gt=gt.detach().cpu().numpy(),
            con_at=at.detach().cpu().numpy(),
            con_sq=sq.detach().cpu().numpy(),
            con_aug_type=np.asarray(aug_type),
            con_aug_val=np.asarray(aug_val, dtype=np.float32))
    hook = model.bbox_head.loss_ss.register_forward_pre_hook(_capture_consistency)
    losses = model.loss(image, [ds])
    hook.remove()
    total = sum(v.sum() for v in losses.values())
    total.backward()
    named = dict(model.named_parameters())
    out = {f'loss__{k}': np.float64(v.sum().detach().cpu())
           for k, v in losses.items()}
    out.update(consistency_inputs)
    if args.dump_generated_cache:
        patterns = model.copy_paste_cache[0]
        out['cp_count'] = np.int32(len(patterns))
        for i, (chip, bbox, label) in enumerate(patterns):
            out[f'cp_chip__{i}'] = chip.detach().cpu().numpy()
            out[f'cp_bbox__{i}'] = np.asarray(bbox)
            out[f'cp_label__{i}'] = np.int32(label)
    for name in WATCH:
        g = named[name].grad
        out[f'grad__{name}'] = g.detach().cpu().numpy()
        print(name, 'norm=', float(g.norm()))
    for k, v in losses.items():
        print(k, float(v.sum()))
    np.savez(args.out, **out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
