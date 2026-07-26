"""dump_golden_p2rv2_loss.py — p2r-torch 环境 dump Point2RBox-v2 四个 loss 的 golden。

按 M3 移植顺序逐段追加：
    §1 GaussianOverlapLoss（含 gwd_sigma_loss / bhattacharyya）
    §2 VoronoiWatershedLoss（后续追加）
    §3 EdgeLoss（后续追加）
    §4 Point2RBoxV2ConsistencyLoss（后续追加）

随机数策略（PLAN §7）：输入全部由固定 seed 的 numpy 生成并存进 npz，
两边从同一 npz 读，不依赖框架内 RNG。
"""
import os
import sys

import numpy as np
import torch

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
sys.path.insert(0, REPO)

OUT = os.path.join(os.path.dirname(__file__), '..', 'tests', 'parity', 'golden',
                   'p2rv2_loss.npz')


def make_gaussians(n, seed):
    """mu (N,2) + sigma (N,2,2)，含 w==h 正方形与部分高重叠对。"""
    rng = np.random.RandomState(seed)
    mu = rng.uniform(100, 900, (n, 2)).astype(np.float32)
    mu[1] = mu[0] + rng.uniform(-6, 6, 2)   # 制造高重叠对
    mu[3] = mu[2] + rng.uniform(-4, 4, 2)
    sigmas = []
    for i in range(n):
        w, h = rng.uniform(8, 120, 2)
        if i == 0:
            w = h = 48.0                     # w==h 正方形
        t = rng.uniform(-np.pi / 2, np.pi / 2)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        sigmas.append(R @ np.diag([(w / 2) ** 2, (h / 2) ** 2]) @ R.T)
    return mu, np.stack(sigmas).astype(np.float32)


def main():
    from mmrotate.models.losses.point2rbox_v2_loss import (
        GaussianOverlapLoss, gwd_sigma_loss)

    out = {}

    # ---- §1 GaussianOverlapLoss（官方 config: loss_weight=10.0, lamb=0；
    #        另测默认 lamb=1e-4 路径）----
    mu_np, sigma_np = make_gaussians(12, 21)
    out['gol_mu'], out['gol_sigma'] = mu_np, sigma_np
    for tag, kwargs in [('cfg', dict(loss_weight=10.0, lamb=0)),
                        ('lamb', dict(loss_weight=1.0, lamb=1e-4))]:
        loss_fn = GaussianOverlapLoss(**kwargs)
        mu = torch.tensor(mu_np, requires_grad=True)
        sigma = torch.tensor(sigma_np, requires_grad=True)
        loss = loss_fn((mu, sigma))
        loss.backward()
        out[f'gol_{tag}_loss'] = loss.item()
        out[f'gol_{tag}_mu_grad'] = mu.grad.detach().numpy()
        out[f'gol_{tag}_sigma_grad'] = sigma.grad.detach().numpy()

    # gwd_sigma_loss 单测（Voronoi 依赖）
    _, sig_a = make_gaussians(8, 22)
    _, sig_b = make_gaussians(8, 23)
    sa = torch.tensor(sig_a, requires_grad=True)
    l = gwd_sigma_loss(sa, torch.tensor(sig_b), reduction='mean')
    l.backward()
    out['gws_a'], out['gws_b'] = sig_a, sig_b
    out['gws_loss'] = l.item()
    out['gws_grad'] = sa.grad.detach().numpy()

    np.savez(OUT, **out)
    print('saved ->', OUT)
    print('gol_cfg_loss =', out['gol_cfg_loss'], '| gol_lamb_loss =', out['gol_lamb_loss'])
    print('gws_loss =', out['gws_loss'])


if __name__ == '__main__':
    main()
