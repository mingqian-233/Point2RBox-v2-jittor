"""Compare one deterministic Jittor train step with official PyTorch dump."""
import argparse
import math

import cv2
import jittor as jt
import numpy as np

from jdet.config import init_cfg
from jdet.runner import Runner
from jdet.data.p2rv2_dota import poly2rbox_le90_np

CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter')


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


def ann(path):
    polys, labels = [], []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) >= 9 and p[8] in CLASSES:
                polys.append([float(x) for x in p[:8]])
                labels.append(CLASSES.index(p[8]))
    boxes = poly2rbox_le90_np(np.asarray(polys, np.float32))
    boxes[:, 2:4], boxes[:, 4] = 1.0, 0.0
    return boxes, np.asarray(labels, np.int32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--golden', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--ann', required=True)
    p.add_argument('--ss-mode', choices=('rot', 'flp', 'sca'), default='rot')
    p.add_argument('--ss-value', type=float, default=None,
                   help='Rotation in degrees or scale factor; defaults to 67.5/1.25.')
    p.add_argument('--epoch', type=int, default=1)
    p.add_argument('--fixed-copy-paste', action='store_true')
    p.add_argument('--compare-generated-cache', action='store_true')
    args = p.parse_args()
    jt.flags.use_cuda = 1
    init_cfg(args.config)
    runner = Runner()
    runner.load(args.checkpoint, model_only=True)
    model = runner.model
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
    rgb = bgr[:, :, ::-1].copy().transpose(2, 0, 1).astype(np.float32)
    mean = np.asarray([123.675, 116.28, 103.53], np.float32)[:, None, None]
    std = np.asarray([58.395, 57.12, 57.375], np.float32)[:, None, None]
    image = jt.array(((rgb - mean) / std)[None])
    boxes, labels = ann(args.ann)
    target = dict(rboxes=jt.array(boxes), labels=jt.array(labels))
    if args.fixed_copy_paste:
        yy, xx = np.mgrid[:20, :24].astype(np.float32)
        pattern = np.stack((
            (xx - 12.0) / 12.0,
            (yy - 10.0) / 10.0,
            (xx + yy - 22.0) / 22.0,
            np.full_like(xx, 0.75)), axis=0).astype(np.float32)
        patch_box = np.asarray([12.0, 10.0, 16.0, 8.0, 0.2], np.float32)
        model.copy_paste_cache = [[(jt.array(pattern), patch_box.copy(), 4)]]
        np.random.seed(314159)
    np.random.seed(314159)
    losses = model(image, [target])
    total = sum(v.sum() for v in losses.values())
    named = dict(model.named_parameters())
    params = [named[k] for k in WATCH]
    grads = jt.grad(total, params)
    golden = np.load(args.golden)
    if args.compare_generated_cache:
        patterns = model.copy_paste_cache[0]
        print('copy_paste_count J=', len(patterns),
              'T=', int(golden['cp_count']))
        for i, (chip, bbox, label) in enumerate(patterns):
            want_chip = golden[f'cp_chip__{i}']
            got_chip = chip.numpy()
            rgb_rel = np.linalg.norm(got_chip[:3] - want_chip[:3]) / max(
                np.linalg.norm(want_chip[:3]), 1e-12)
            print('copy_paste', i, 'shape J/T=', got_chip.shape,
                  want_chip.shape, 'label J/T=', label,
                  int(golden[f'cp_label__{i}']),
                  'bbox J=', bbox, 'T=', golden[f'cp_bbox__{i}'],
                  'rgb_rel=', rgb_rel,
                  'alpha_mean J/T=', got_chip[3].mean(),
                  want_chip[3].mean())
    if 'con_go' in golden:
        from jdet.models.losses.point2rbox_v2_loss import Point2RBoxV2ConsistencyLoss
        raw_loss = Point2RBoxV2ConsistencyLoss(loss_weight=1.0)(
            (jt.array(golden['con_go']), jt.array(golden['con_ao'])),
            (jt.array(golden['con_gt']), jt.array(golden['con_at'])),
            jt.array(golden['con_sq']),
            str(golden['con_aug_type']),
            float(golden['con_aug_val']))
        want = float(golden['loss__loss_ss'])
        print('loss_ss_on_exact_torch_inputs J=', float(raw_loss),
              'T=', want, 'rel=', abs(float(raw_loss) - want) / max(abs(want), 1e-12))
    for k, v in losses.items():
        got, want = float(v.sum()), float(golden[f'loss__{k}'])
        print(k, 'J=', got, 'T=', want,
              'rel=', abs(got - want) / max(abs(want), 1e-12))
    for name, g in zip(WATCH, grads):
        got, want = g.numpy(), golden[f'grad__{name}']
        rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12)
        print(name, 'Jnorm=', np.linalg.norm(got),
              'Tnorm=', np.linalg.norm(want), 'rel_l2=', rel)


if __name__ == '__main__':
    main()
