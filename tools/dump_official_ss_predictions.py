"""Dump official PyTorch self-supervision predictions for cross-framework audit.

Run in the p2r-torch environment with the Point2RBox-v3 source on PYTHONPATH.
The sample and augmentation constants match visualize_ss_predictions.py.
"""
import argparse
import copy
import math
import os

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from torchvision.transforms import functional as TVF


CLASSES = (
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
    'harbor', 'swimming-pool', 'helicopter')


def load_ann(path):
    boxes, labels = [], []
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 9 or p[8] not in CLASSES:
                continue
            q = np.asarray(p[:8], np.float32).reshape(4, 2)
            (x, y), (w, h), a = cv2.minAreaRect(q)
            boxes.append([x, y, w, h, math.radians(a)])
            labels.append(CLASSES.index(p[8]))
    return np.asarray(boxes, np.float32), np.asarray(labels, np.int64)


def sample(boxes, labels):
    ds = DetDataSample(
        metainfo=dict(img_shape=(1024, 1024), ori_shape=(1024, 1024),
                      scale_factor=(1.0, 1.0)))
    ds.gt_instances = InstanceData(
        bboxes=RotatedBoxes(torch.from_numpy(boxes).cuda()),
        labels=torch.from_numpy(labels).cuda())
    return ds


def covariance(boxes):
    wh, a = boxes[:, 2:4] * .5, boxes[:, 4]
    c, s = np.cos(a), np.sin(a)
    r = np.stack([c, -s, s, c], 1).reshape(-1, 2, 2)
    d = np.zeros_like(r)
    d[:, 0, 0], d[:, 1, 1] = wh[:, 0], wh[:, 1]
    g = r @ d
    return g @ np.transpose(g, (0, 2, 1))


def discrepancy(a, b):
    ca, cb = covariance(a), covariance(b)
    scale = np.maximum(np.sqrt((ca * ca).sum((1, 2))), 1e-6)
    sigma = np.sqrt(((ca - cb) ** 2).sum((1, 2))) / scale
    angle = np.abs((b[:, 4] - a[:, 4] + np.pi / 2) % np.pi - np.pi / 2)
    return sigma, angle


def inverse_boxes(boxes, aug, value, h, w):
    b = boxes.copy()
    if aug == 'rot':
        c, s = math.cos(value), math.sin(value)
        ctr = np.float32([w / 2, h / 2])
        b[:, :2] = (b[:, :2] - ctr) @ np.float32([[c, -s], [s, c]]) + ctr
        b[:, 4] -= value
    elif aug == 'flp':
        b[:, 1] = h - b[:, 1]
        b[:, 4] = -b[:, 4]
    else:
        b[:, :4] /= value
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--ann', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    init_default_scope('mmrotate')
    cfg = Config.fromfile(args.config)
    model = MODELS.build(cfg.model).cuda().eval()
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.bbox_head.train()  # GT-point pseudo prediction path

    bgr = cv2.imread(args.image)
    rgb = torch.from_numpy(bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().cuda()
    mean = torch.tensor([123.675, 116.28, 103.53], device='cuda')[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375], device='cuda')[:, None, None]
    image = ((rgb - mean) / std)[None]
    boxes, labels = load_ann(args.ann)
    base = sample(boxes, labels)
    h, w = image.shape[-2:]

    outputs = {}
    for aug, value in [('rot', math.radians(67.5)), ('flp', 0.0),
                       ('sca', 0.70)]:
        aug_ds = copy.deepcopy(base)
        if aug == 'rot':
            aug_im, inst = model.rotate_crop(
                image, value, (h, w), [aug_ds.gt_instances], 'reflection')
            aug_ds.gt_instances = inst[0]
        elif aug == 'flp':
            aug_im = TVF.vflip(image)
            aug_ds.gt_instances.bboxes.flip_([h, w], 'vertical')
        else:
            aug_im = TVF.resized_crop(
                image, 0, 0, int(h / value), int(w / value), [h, w],
                antialias=True)
            aug_ds.gt_instances.bboxes.rescale_([value, value])
        feat = model.extract_feat(torch.cat([image, aug_im]))
        pred = model.bbox_head.predict(feat, [base, aug_ds])
        ori = pred[0].bboxes.tensor.detach().cpu().numpy()
        trs = pred[1].bboxes.tensor.detach().cpu().numpy()
        inv = inverse_boxes(trs, aug, value, h, w)
        sig, ang = discrepancy(ori, inv)
        outputs[f'{aug}_ori'] = ori
        outputs[f'{aug}_trs'] = trs
        outputs[f'{aug}_inv'] = inv
        outputs[f'{aug}_sigma'] = sig
        outputs[f'{aug}_angle'] = ang
        print(f'{aug}: median sigma={np.median(sig):.6f}, '
              f'median angle={np.degrees(np.median(ang)):.3f}deg, '
              f'p95 sigma={np.percentile(sig,95):.6f}, '
              f'p95 angle={np.degrees(np.percentile(ang,95)):.3f}deg')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **outputs)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
