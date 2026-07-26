"""Point2RBox-v2 的四个 loss（Jittor 移植）。

源：/root/ref/Point2RBox-v3/mmrotate/models/losses/point2rbox_v2_loss.py（v3 版，
含对 v2 向后兼容的扩展参数，默认值即 v2 行为——与 Agent B 的约定）。

移植要点（详见 docs/porting_notes.md）：
- torch.linalg.eigh/solve → jdet.ops.linalg2x2 的 2×2 闭式解（可导、退化保护）
- loss[torch.eye(B)] = 0 等 in-place 写法 → out-of-place（乘 mask / jt.where），
  Jittor 的 in-place 自动微分不可靠（前向对、梯度 0）
- 上游源文件顶部 3 处 IDE 误加 import（click/pandas/sympy）按计划 §8 删除
"""
import math

import cv2
import numpy as np
import jittor as jt
from jittor import nn

from jdet.utils.registry import LOSSES
from jdet.models.losses.gaussian_dist_loss import postprocess, weight_reduce_loss
from jdet.ops.linalg2x2 import eigh_2x2, solve_2x2, diag_embed_2x2


def gwd_sigma_loss(pred, target, weight=None, fun='log1p', tau=1.0, alpha=1.0,
                   normalize=True, reduction='mean', avg_factor=None):
    """GWD loss 的仅 sigma 版本（忽略 mu）。对齐上游 @weighted_loss 展开形式。"""
    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p[..., 0, 0] + Sigma_p[..., 1, 1]
    whr_distance = whr_distance + Sigma_t[..., 0, 0] + Sigma_t[..., 1, 1]

    _t_tr = jt.matmul(Sigma_p, Sigma_t)
    _t_tr = _t_tr[..., 0, 0] + _t_tr[..., 1, 1]
    det_p = Sigma_p[..., 0, 0] * Sigma_p[..., 1, 1] - Sigma_p[..., 0, 1] * Sigma_p[..., 1, 0]
    det_t = Sigma_t[..., 0, 0] * Sigma_t[..., 1, 1] - Sigma_t[..., 0, 1] * Sigma_t[..., 1, 0]
    _t_det_sqrt = (det_p * det_t).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * ((_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (_t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    loss = postprocess(distance, fun=fun, tau=tau)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


def bhattacharyya_coefficient(pred, target):
    """2-D 高斯分布间的 Bhattacharyya 系数，shape (N,)（batch 维保留）。"""
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_M = (Sigma_p + Sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    t0 = jt.exp(-0.125 * jt.matmul(dxy.permute(0, 2, 1), solve_2x2(Sigma_M, dxy)))
    det_p = Sigma_p[:, 0, 0] * Sigma_p[:, 1, 1] - Sigma_p[:, 0, 1] * Sigma_p[:, 1, 0]
    det_t = Sigma_t[:, 0, 0] * Sigma_t[:, 1, 1] - Sigma_t[:, 0, 1] * Sigma_t[:, 1, 0]
    det_m = Sigma_M[:, 0, 0] * Sigma_M[:, 1, 1] - Sigma_M[:, 0, 1] * Sigma_M[:, 1, 0]
    t1 = (det_p * det_t).clamp(1e-7).sqrt()
    t2 = det_m

    coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
    coef = coef.reshape(_shape[:-1])
    return coef


def gaussian_overlap_loss(pred, target, weight=None, alpha=0.01, beta=0.6065,
                          overlap_scale=None, reduction='mean', avg_factor=None):
    """基于 Bhattacharyya 系数的高斯重叠 loss（@weighted_loss 展开形式）。"""
    mu, sigma = pred
    B = mu.shape[0]
    mu0 = mu[None].expand(B, B, 2)
    sigma0 = sigma[None].expand(B, B, 2, 2)
    mu1 = mu[:, None].expand(B, B, 2)
    sigma1 = sigma[:, None].expand(B, B, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    if overlap_scale is not None:
        loss = jt.multiply(loss, overlap_scale) * overlap_scale.numel() / nn.relu(overlap_scale).sum()

    # 上游 loss[torch.eye(B, dtype=bool)] = 0 → out-of-place（对角元乘 0，梯度等价）
    loss = loss * (1 - jt.init.eye(B, dtype=loss.dtype))
    loss = nn.leaky_relu(loss - beta, scale=alpha) + beta * alpha
    loss = loss.sum(-1)
    return weight_reduce_loss(loss, weight, reduction, avg_factor)


@LOSSES.register_module()
class GaussianOverlapLoss(nn.Module):
    """Gaussian Overlap Loss（官方 config：loss_weight=10.0, lamb=0）。"""

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 lamb=1e-4):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb

    def execute(self,
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                overlap_scale=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])

        mu, sigma = pred

        L = eigh_2x2(sigma)[0].clamp(1e-7).sqrt()
        # F.l1_loss(L, zeros, reduction='none') == |L|
        loss_lamb = jt.abs(L)
        loss_lamb = self.lamb * jt.log(1 + loss_lamb).mean()

        overlap_loss = gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            overlap_scale=overlap_scale,
        )

        return self.loss_weight * (loss_lamb + overlap_loss)
