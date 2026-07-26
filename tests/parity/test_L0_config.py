"""L0 配置 parity：Jittor config ↔ 官方 config golden 逐值零容差比对。

golden 由 tools/dump_config.py 在 p2r-torch 环境生成（296/297/246 个扁平键）。
Jittor 侧 config（configs/point2rbox_v2/*.py）在 M4 编写；写好后在
CONFIG_PAIRS 里登记「Jittor config 路径 + 键映射函数」，测试自动逐值比对。

映射规则（铁律一允许变的三类）在 to_official_key()/normalize_value() 中集中声明，
其余任何键值差异都是 FAIL——数值、布尔、pipeline 顺序、列表长度零容差。
"""
import json
import os

import pytest

GOLDEN = os.path.join(os.path.dirname(__file__), 'golden')

# (golden json, jittor config path, 适配器)；M4 起逐个填充
CONFIG_PAIRS = [
    # ('config_point2rbox_v2-1x-dota.json', 'configs/point2rbox_v2/point2rbox_v2_1x_dota.py', ...),
]


def load_golden(name):
    with open(os.path.join(GOLDEN, name)) as f:
        return json.load(f)


class TestGoldenIntegrity:
    """golden 自身的完整性 + 铁律二关键值锚定（防 golden 被误重dump 后漂移）。"""

    def test_iron_rules_anchored(self):
        d = load_golden('config_point2rbox_v2-1x-dota.json')
        assert d['optim_wrapper.clip_grad.max_norm'] == 35            # 铁律二 #1
        assert d['optim_wrapper.clip_grad.norm_type'] == 2
        assert d['param_scheduler[0].type'] == 'LinearLR'             # 铁律二 #2
        assert d['param_scheduler[0].start_factor'] == pytest.approx(1 / 3)
        assert d['param_scheduler[0].end'] == 500
        assert d['param_scheduler[0].by_epoch'] is False
        assert d['param_scheduler[1].type'] == 'MultiStepLR'
        assert d['param_scheduler[1].milestones[0]'] == 8
        assert d['param_scheduler[1].milestones[1]'] == 11
        assert d['custom_hooks[0].type'] == 'mmdet.SetEpochInfoHook'  # 铁律二 #3
        assert d['model.bbox_head.strides.__len__'] == 1              # 铁律二 #6
        assert d['model.bbox_head.strides[0]'] == 8
        assert d['model.bbox_head.square_cls[0]'] == 1                # 铁律二 #8
        assert d['model.bbox_head.square_cls[1]'] == 9
        assert d['model.bbox_head.square_cls[2]'] == 11
        assert d['model.ss_prob[0]'] == 0.68                          # 铁律二 #9（挂在 model 下）
        assert d['model.ss_prob[1]'] == 0.07
        assert d['model.ss_prob[2]'] == 0.25
        assert d['optim_wrapper.optimizer.weight_decay'] == 0.05
        assert d['optim_wrapper.optimizer.lr'] == 5e-05
        assert d['train_cfg.max_epochs'] == 12
        assert d['train_dataloader.batch_size'] == 2

    def test_stage2_iron_rules(self):
        d = load_golden('config_rotated-fcos-1x-dota-using-pseudo.json')
        assert d['optim_wrapper.optimizer.weight_decay'] == 0.005     # 铁律二 #5：就是差 10 倍
        assert d['model.backbone.out_indices[0]'] == 0                # 铁律二 #7
        assert d['model.backbone.out_indices.__len__'] == 4
        assert d['model.neck.out_channels'] == 512
        assert d['train_dataloader.batch_size'] == 4

    def test_val_points_to_trainval(self):
        d = load_golden('config_point2rbox_v2-1x-dota.json')
        # 铁律二 #4：val 指向 trainval/，照抄
        assert 'trainval' in d['val_dataloader.dataset.ann_file']


@pytest.mark.skipif(not CONFIG_PAIRS, reason='Jittor 侧 config 尚未编写（M4）')
class TestConfigParity:
    def test_pointwise(self):
        raise NotImplementedError  # M4 实现：加载 Jittor config → 映射 → 与 golden 全键零容差
