import os, sys
import numpy as np
sys.path.insert(0, '/root/work/A/Point2RBox-v2-jittor/python')
import jittor as jt
GOLDEN = '/root/work/A/Point2RBox-v2-jittor/tests/parity/golden'

def run(cuda):
    jt.flags.use_cuda = cuda
    from jdet.models.roi_heads.point2rbox_v2_head import Point2RBoxV2Head
    g = np.load(os.path.join(GOLDEN, 'head_parity.npz'))
    head = Point2RBoxV2Head(
        num_classes=15, in_channels=128, feat_channels=128, strides=[8],
        square_cls=[1, 9, 11],
        edge_loss_cls=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13],
        post_process={11: 1.2},
        voronoi_type='standard',
        voronoi_thres=dict(default=[0.994, 0.005],
                           override=(([2, 11], [0.999, 0.6]),
                                     ([7, 8, 10, 14], [0.95, 0.005]))),
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0),
        loss_overlap=dict(type='GaussianOverlapLoss', loss_weight=10.0, lamb=0))
    head.epoch = 1
    head.train()
    head.load_parameters({k[3:]: jt.array(g[k]) for k in g.files if k.startswith('w__')})
    head.images = jt.array(g['images'])
    targets = [dict(rboxes=jt.array(g[f'gt{i}_rb']),
                    labels=jt.array(g[f'gt{i}_lb'].astype(np.int32)),
                    bids=jt.array(g[f'gt{i}_bid'].astype(np.int32)),
                    ss=('rot', float(g['ss_val']))) for i in range(4)]
    feat = jt.array(g['feat'])
    losses = head.loss([feat], targets)
    lv = {k: float(v.sum().item()) for k, v in losses.items()}
    total = sum(v.sum() for v in losses.values())
    grad = jt.grad(total, feat).numpy()
    return lv, grad

lv_g, grad_gpu = run(1)
lv_c, grad_cpu = run(0)
print('losses GPU:', {k: round(v,6) for k,v in lv_g.items()})
print('losses CPU:', {k: round(v,6) for k,v in lv_c.items()})
r = np.linalg.norm(grad_gpu-grad_cpu)/(np.linalg.norm(grad_cpu)+1e-12)
scale = np.abs(grad_cpu).max()
viol = (np.abs(grad_gpu-grad_cpu) > 5e-2*np.maximum(np.abs(grad_cpu), scale*5e-2)).mean()
print(f'GPU vs CPU grad: rel_l2={r:.3e} viol={viol:.4f}')
