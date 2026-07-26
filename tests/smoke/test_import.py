"""冒烟：核心包可 import、numpy 版本未被顶到 2.x（docs/environment.md 坑 #1）。"""


def test_numpy_version():
    import numpy
    major = int(numpy.__version__.split('.')[0])
    assert major < 2, (
        f'numpy=={numpy.__version__}：jittor 1.3.8.5 + numpy>=2 会静默产出错误数值，'
        '必须 pip install numpy==1.26.4（见 docs/environment.md 坑 #1）')


def test_jittor_numerics():
    import jittor as jt
    a = jt.float32([1, 2, 3])
    assert (a + a).numpy().tolist() == [2., 4., 6.], \
        'jittor elementwise 输出错误——大概率是 numpy 2.x 兼容问题'


def test_jdet_imports():
    import jdet  # noqa: F401
    from jdet.optims.optimizer import AdamW  # noqa: F401
    from jdet.optims.lr_scheduler import LinearWarmupMultiStepLR  # noqa: F401
    from jdet.runner import Runner  # noqa: F401
