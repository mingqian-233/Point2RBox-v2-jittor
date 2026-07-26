"""L2 模块级 parity（M2.5 复用件核对起步）。

golden 来源：
    gdloss_gwd.npz  <- tools/dump_golden_gdloss.py（p2r-torch 环境，mmrotate 实现）

容差（PLAN §7）：前向 rtol 1e-4，梯度 rtol 1e-3（梯度必须比——防 in-place 静默断梯度）。
"""
import os

import numpy as np
import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


def _assert_close(a, b, rtol, name=''):
    """rtol 逐元素相对容差 + 按张量量级缩放的绝对容差。

    纯逐元素相对误差对「代数上恰好为 0、浮点上有舍入」的元素会爆炸
    （例：w==h 时 sigma 非对角元 golden=0.0、jittor=1.3e-5，而矩阵量级 1e3），
    因此 atol = rtol * max|golden|，即把近零元素的误差换算到张量量级上衡量。"""
    scale = float(np.max(np.abs(b))) or 1.0
    np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol * scale, err_msg=name)


def _rel_err(a, b):
    denom = np.maximum(np.abs(b), 1e-12)
    return np.max(np.abs(a - b) / denom)


class TestGDLossGWD:
    """底座 gaussian_dist_loss.py vs mmrotate 官方（含退化用例：w==h、±π/2、极小框、大长宽比）。"""

    @pytest.fixture(scope='class')
    def golden(self):
        return np.load(os.path.join(GOLDEN, 'gdloss_gwd.npz'))

    def test_xy_wh_r_2_xy_sigma_forward(self, golden):
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma

        pred = jt.array(golden['pred'])
        xy, sigma = xy_wh_r_2_xy_sigma(pred)
        _assert_close(xy.numpy(), golden['xy'], 1e-4, 'xy')
        _assert_close(sigma.numpy(), golden['sigma'], 1e-4, 'sigma')

    def test_xy_wh_r_2_xy_sigma_grad(self, golden):
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import xy_wh_r_2_xy_sigma

        pred = jt.array(golden['pred'])
        xy, sigma = xy_wh_r_2_xy_sigma(pred)
        g = jt.grad(xy.sum() + sigma.sum(), pred)
        _assert_close(g.numpy(), golden['xy_sigma_grad'], 1e-3, 'xy_sigma_grad')

    def test_gwd_loss_forward(self, golden):
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import GDLoss

        loss_fn = GDLoss(loss_type='gwd', loss_weight=5.0)
        loss = loss_fn(jt.array(golden['pred']), jt.array(golden['target']))
        rel = abs(float(loss.item()) - float(golden['gwd_loss'])) / abs(float(golden['gwd_loss']))
        assert rel < 1e-4, f'forward rel err = {rel}'

    def test_gwd_loss_grad(self, golden):
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import GDLoss

        loss_fn = GDLoss(loss_type='gwd', loss_weight=5.0)
        pred = jt.array(golden['pred'])
        loss = loss_fn(pred, jt.array(golden['target']))
        g = jt.grad(loss, pred)
        gn = g.numpy()
        assert np.abs(gn).sum() > 0, '梯度全 0——疑似 in-place 断梯度'
        _assert_close(gn, golden['gwd_grad'], 1e-3, 'gwd_grad')

    def test_gwd_loss_weighted(self, golden):
        """带 weight + avg_factor 的路径（head 实际调用形态）：前向 + 梯度。"""
        import jittor as jt
        from jdet.models.losses.gaussian_dist_loss import GDLoss

        loss_fn = GDLoss(loss_type='gwd', loss_weight=5.0)
        pred = jt.array(golden['pred'])
        loss = loss_fn(pred, jt.array(golden['target']),
                       weight=jt.array(golden['weight']),
                       avg_factor=float(golden['avg_factor']))
        rel = abs(float(loss.item()) - float(golden['gwd_loss_weighted'])) \
            / abs(float(golden['gwd_loss_weighted']))
        assert rel < 1e-4, f'weighted forward rel err = {rel}'
        g = jt.grad(loss, pred)
        _assert_close(g.numpy(), golden['gwd_grad_weighted'], 1e-3, 'gwd_grad_weighted')


class TestGaussianOverlapLoss:
    """M3 §1：GaussianOverlapLoss vs mmrotate（官方参数 w=10,lamb=0 + 默认 lamb 路径）。"""

    @pytest.fixture(scope='class')
    def g(self):
        return np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))

    def _run(self, g, loss_weight, lamb, tag):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import GaussianOverlapLoss

        mu = jt.array(g['gol_mu'])
        sigma = jt.array(g['gol_sigma'])
        loss = GaussianOverlapLoss(loss_weight=loss_weight, lamb=lamb)((mu, sigma))
        rel = abs(float(loss.item()) - float(g[f'gol_{tag}_loss'])) \
            / abs(float(g[f'gol_{tag}_loss']))
        assert rel < 1e-4, f'{tag} forward rel err = {rel}'
        gm, gs = jt.grad(loss, [mu, sigma])
        assert np.abs(gm.numpy()).sum() > 0, 'mu 梯度全 0'
        _assert_close(gm.numpy(), g[f'gol_{tag}_mu_grad'], 1e-3, f'{tag}_mu_grad')
        _assert_close(gs.numpy(), g[f'gol_{tag}_sigma_grad'], 1e-3, f'{tag}_sigma_grad')

    def test_official_cfg(self, g):
        self._run(g, 10.0, 0, 'cfg')

    def test_default_lamb(self, g):
        self._run(g, 1.0, 1e-4, 'lamb')


class TestGwdSigmaLoss:
    def test_forward_grad(self):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import gwd_sigma_loss

        g = np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))
        a = jt.array(g['gws_a'])
        loss = gwd_sigma_loss(a, jt.array(g['gws_b']), reduction='mean')
        rel = abs(float(loss.item()) - float(g['gws_loss'])) / abs(float(g['gws_loss']))
        assert rel < 1e-4, f'forward rel err = {rel}'
        grad = jt.grad(loss, a)
        _assert_close(grad.numpy(), g['gws_grad'], 1e-3, 'gws_grad')


class TestVoronoiWatershedLoss:
    """M3 §2：官方路径 voronoi='standard', w=5.0（watershed 走 cv2，两边共享）。"""

    def test_forward_grad_and_markers(self):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import VoronoiWatershedLoss

        g = np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))
        loss_fn = VoronoiWatershedLoss(loss_weight=5.0)
        mu = jt.array(g['vws_mu'])
        sigma = jt.array(g['vws_sigma'])
        loss = loss_fn((mu, sigma), jt.array(g['vws_label']), jt.array(g['vws_image']),
                       jt.array(g['vws_pos']), jt.array(g['vws_neg']),
                       voronoi='standard')
        # markers（watershed 结果）应逐像素一致（输入图完全相同时 cv2 确定性）
        markers = loss_fn.vis[1].numpy()
        mismatch = (markers != g['vws_markers']).mean()
        assert mismatch < 0.001, f'markers 不一致像素比例 = {mismatch}'
        rel = abs(float(loss.item()) - float(g['vws_loss'])) / abs(float(g['vws_loss']))
        assert rel < 1e-3, f'forward rel err = {rel}'
        gs = jt.grad(loss, sigma).numpy()
        assert np.abs(gs).sum() > 0, 'sigma 梯度全 0'
        # w==h 各向同性行：eigh 特征基任意 → 逐特征值子梯度在 a/c 间的分配不唯一
        # （torch 选 I 基），只比基无关的 trace；非简并行逐位比
        sg = g['vws_sigma']
        tr = sg[:, 0, 0] + sg[:, 1, 1]
        disc = np.sqrt(((sg[:, 0, 0] - sg[:, 1, 1]) / 2) ** 2 + sg[:, 0, 1] ** 2)
        nondeg = disc / (tr / 2) > 1e-3
        want = g['vws_sigma_grad']
        _assert_close(gs[nondeg], want[nondeg], 1e-2, 'vws_sigma_grad(nondeg)')
        np.testing.assert_allclose(gs[~nondeg, 0, 0] + gs[~nondeg, 1, 1],
                                   want[~nondeg, 0, 0] + want[~nondeg, 1, 1],
                                   rtol=1e-2, err_msg='vws_sigma_grad(trace)')


class TestEdgeLoss:
    def test_forward_grad(self):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import EdgeLoss

        g = np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))
        jt.flags.use_cuda = 1  # RoIAlignRotated 只有 CUDA 实现
        loss_fn = EdgeLoss(loss_weight=0.3)
        b = jt.array(g['edge_boxes'])
        loss = loss_fn([b], jt.array(g['edge_map']))
        rel = abs(float(loss.item()) - float(g['edge_loss'])) / abs(float(g['edge_loss']))
        grad = jt.grad(loss, b)
        jt.flags.use_cuda = 0
        assert rel < 1e-3, f'forward rel err = {rel}'
        assert np.abs(grad.numpy()).sum() > 0, '梯度全 0'
        _assert_close(grad.numpy(), g['edge_grad'], 1e-2, 'edge_grad')


class TestConsistencyLoss:
    def _run(self, aug_type, aug_val):
        import jittor as jt
        from jdet.models.losses.point2rbox_v2_loss import Point2RBoxV2ConsistencyLoss

        g = np.load(os.path.join(GOLDEN, 'p2rv2_loss.npz'))
        loss_fn = Point2RBoxV2ConsistencyLoss(loss_weight=1.0)
        go = jt.array(g['con_gaus_o'])
        gt_ = jt.array(g['con_gaus_t'])
        ao = jt.array(g['con_ang_o'])
        at = jt.array(g['con_ang_t'])
        loss = loss_fn((go, ao), (gt_, at), jt.array(g['con_sq']), aug_type, aug_val)
        want = float(g[f'con_{aug_type}_loss'])
        rel = abs(float(loss.item()) - want) / abs(want)
        assert rel < 1e-4, f'{aug_type} forward rel err = {rel}'
        ggo, gao = jt.grad(loss, [go, ao])
        assert np.abs(ggo.numpy()).sum() > 0
        _assert_close(ggo.numpy(), g[f'con_{aug_type}_go_grad'], 1e-3, f'{aug_type}_go_grad')
        _assert_close(gao.numpy(), g[f'con_{aug_type}_ao_grad'], 1e-3, f'{aug_type}_ao_grad')

    def test_rot(self):
        self._run('rot', 0.6)

    def test_flp(self):
        self._run('flp', 0.0)

    def test_sca(self):
        self._run('sca', 1.3)
