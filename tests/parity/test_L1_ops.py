"""L1 算子/组件级 parity 测试（CPU 可跑，进 CI）。

golden 来源：
    lr_sequence.npz  <- tools/dump_lr_mmengine.py（p2r-torch 环境）
"""
import os

import numpy as np
import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')


class TestLRSchedule:
    """C2：LinearWarmupMultiStepLR 与 mmengine LinearLR+MultiStepLR 逐点相等。"""

    def test_lr_sequence_pointwise(self):
        import jittor as jt
        from jdet.optims.optimizer import AdamW
        from jdet.optims.lr_scheduler import LinearWarmupMultiStepLR

        g = np.load(os.path.join(GOLDEN, 'lr_sequence.npz'))
        golden_lr = g['lr']
        ipe, epochs, base_lr = int(g['iters_per_epoch']), int(g['epochs']), float(g['base_lr'])

        w = jt.ones(2)
        opt = AdamW([w], lr=base_lr, betas=(0.9, 0.999), weight_decay=0.05)
        sched = LinearWarmupMultiStepLR(
            opt, start_factor=float(g['start_factor']), warmup_iters=int(g['warmup_iters']),
            milestones=[int(m) for m in g['milestones']], gamma=float(g['gamma']))

        # 模拟 Runner.train 的调用时序：每 iter 更新前 step(iter, epoch)
        lrs = []
        it = 0
        for e in range(epochs):
            for _ in range(ipe):
                sched.step(it, e, by_epoch=True)
                lrs.append(opt.param_groups[0].get('lr', opt.lr))
                it += 1

        lrs = np.array(lrs, dtype=np.float64)
        assert lrs.shape == golden_lr.shape
        # mmengine 侧是乘法递推，带 ~1e-19 浮点累积误差；1e-9 相对容差即"逐点相等"
        np.testing.assert_allclose(lrs, golden_lr, rtol=1e-9, atol=0)

    def test_lr_key_points(self):
        """独立于 golden 的关键点自检（防 golden 本身 dump 错）。"""
        import jittor as jt
        from jdet.optims.optimizer import AdamW
        from jdet.optims.lr_scheduler import LinearWarmupMultiStepLR

        base = 5e-5
        w = jt.ones(2)
        opt = AdamW([w], lr=base)
        sched = LinearWarmupMultiStepLR(opt, start_factor=1 / 3, warmup_iters=500,
                                        milestones=[8, 11], gamma=0.1)
        cases = [
            (0, 0, base / 3),                          # warmup 起点
            (250, 0, base * (1 / 3 + 2 / 3 * 250 / 499)),  # warmup 中段（分母 W-1=499）
            (499, 0, base),                            # warmup 在第 W-1 iter 到顶（mmengine 语义）
            (500, 0, base),                            # 之后保持
            (5000, 7, base),                           # epoch 7 全量
            (5000, 8, base * 0.1),                     # epoch 8 第一档衰减
            (5000, 11, base * 0.01),                   # epoch 11 第二档
        ]
        for it, ep, expect in cases:
            sched.step(it, ep)
            got = opt.param_groups[0].get('lr', opt.lr)
            assert abs(got - expect) < 1e-15, f'iter={it} epoch={ep}: {got} != {expect}'


class TestPSCCoder:
    """底座 PSCCoder vs mmrotate（官方 config 参数：le90, dual_freq=False, num_step=3, thr_mod=0）。"""

    def test_encode_decode(self):
        import jittor as jt
        from jdet.models.boxes.coder import PSCCoder

        g = np.load(os.path.join(GOLDEN, 'ops_misc.npz'))
        coder = PSCCoder(angle_version='le90', dual_freq=False, num_step=3, thr_mod=0)
        assert coder.encode_size == 3
        enc = coder.encode(jt.array(g['psc_angles']))
        np.testing.assert_allclose(enc.numpy(), g['psc_encoded'], rtol=1e-5, atol=1e-6)
        dec = coder.decode(jt.array(g['psc_encoded']), keepdim=True)
        # 角度按模 π 的角距离比较：±π/2 端点处 atan2(∓0,-1) 的符号翻转会给出
        # +π/2 vs -π/2——le90 下同一朝向（周期 π），不是语义差异
        d = np.abs(dec.numpy() - g['psc_decoded'])
        d = np.minimum(d, np.abs(d - np.pi))
        assert d.max() < 1e-5, f'decode 角距离 max = {d.max()}'


class TestBoxIouRotated:
    def test_iou_vs_mmcv(self):
        import jittor as jt
        from jdet.ops.box_iou_rotated import box_iou_rotated

        g = np.load(os.path.join(GOLDEN, 'ops_misc.npz'))
        iou = box_iou_rotated(jt.array(g['iou_boxes1']), jt.array(g['iou_boxes2']))
        np.testing.assert_allclose(iou.numpy(), g['iou'], rtol=1e-4, atol=1e-5)


class TestNMSRotated:
    def test_keep_set_vs_mmcv(self):
        """keep 集合与 mmcv 一致（iou_threshold=0.1，官方 test_cfg 值）。

        已知语义差异（M4 head 层需适配）：jdet 返回原始下标顺序，
        mmcv 返回按分数降序——影响 max_per_img 截断，此处只比集合。"""
        import jittor as jt
        from jdet.ops.nms_rotated import nms_rotated

        g = np.load(os.path.join(GOLDEN, 'ops_misc.npz'))
        keep = nms_rotated(jt.array(g['nms_boxes']), jt.array(g['nms_scores']), 0.1)
        assert sorted(keep.numpy().tolist()) == sorted(g['nms_keep'].tolist())


class TestRoIAlignRotated:
    """B 问的角度约定：jdet ROIAlignRotated（无 clockwise 参数）vs mmcv clockwise=True。"""

    def _run(self, osize):
        import jittor as jt
        from jdet.ops.roi_align_rotated import ROIAlignRotated

        g = np.load(os.path.join(GOLDEN, 'ops_misc.npz'))
        jt.flags.use_cuda = 1  # jdet 该算子只有 CUDA 实现
        # mmcv(clockwise=True) 等价参数：aligned=True + clockwise=True（=JDet 原生方向）
        op = ROIAlignRotated(osize, spatial_scale=1.0, sampling_ratio=2,
                             aligned=True, clockwise=True)
        out = op(jt.array(g['ra_feat']), jt.array(g['ra_rois']))
        jt.flags.use_cuda = 0
        return out.numpy(), g[f'ra_out_{osize}']

    def test_out7_matches_clockwise_true(self):
        got, want = self._run(7)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)

    def test_out49_matches_clockwise_true(self):
        got, want = self._run(49)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


class TestClipGrad:
    """C1：AdamW(grad_clip=dict(max_norm=35, norm_type=2)) 裁剪后全局 L2 范数 == 35。"""

    def test_clip_grad_norm_35(self):
        import jittor as jt
        from jdet.optims.optimizer import AdamW

        w1 = jt.ones(10)
        w2 = jt.ones(5)
        opt = AdamW([w1, w2], lr=5e-5, grad_clip=dict(max_norm=35, norm_type=2))
        # loss = 100*sum(w1) + 200*sum(w2) → grad w1 全 100、w2 全 200
        # 全局 L2 = sqrt(10*100^2 + 5*200^2) = sqrt(300000) ≈ 547.7 > 35 → 触发裁剪
        loss = (w1 * 100).sum() + (w2 * 200).sum()
        opt.pre_step(loss)
        sq = 0.0
        for pg in opt.param_groups:
            for p, g in zip(pg['params'], pg['grads']):
                if p.is_stop_grad():
                    continue
                sq += float((g * g).sum())
        assert abs(np.sqrt(sq) - 35.0) < 1e-3, f'clipped norm = {np.sqrt(sq)}'
        opt.zero_grad()

    def test_no_clip_when_below(self):
        import jittor as jt
        from jdet.optims.optimizer import AdamW

        w = jt.ones(4)
        opt = AdamW([w], lr=5e-5, grad_clip=dict(max_norm=35, norm_type=2))
        loss = w.sum()  # grad 全 1，L2 = 2 < 35 → 不裁剪
        opt.pre_step(loss)
        g = opt.param_groups[0]['grads'][0]
        np.testing.assert_allclose(g.numpy(), np.ones(4), rtol=1e-6)
        opt.zero_grad()
