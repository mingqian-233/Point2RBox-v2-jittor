import numpy as np, sys
sys.path.insert(0, '/root/work/A/Point2RBox-v2-jittor/python')
import jittor as jt

def group_mean_scatter(values, idx, G):
    N, C = values.shape
    cnt = jt.zeros((G, 1))
    cnt = cnt.scatter(0, idx.unsqueeze(-1), jt.ones((N, 1)), reduce='add')
    s = jt.zeros((G, C))
    s = s.scatter(0, idx.unsqueeze(-1).expand((N, C)), values, reduce='add')
    return s / cnt.clamp(1)

def group_mean_onehot(values, idx, G):
    N, C = values.shape
    oh = (idx.unsqueeze(0) == jt.arange(G).unsqueeze(-1)).float()  # (G,N)
    cnt = oh.sum(1, keepdims=True)
    return jt.matmul(oh, values) / cnt.clamp(1)

rng = np.random.RandomState(0)
N, C, G = 5000, 8, 300
v_np = rng.randn(N, C).astype(np.float32)
i_np = rng.randint(0, G, N).astype(np.int32)

for cuda in (0, 1):
    jt.flags.use_cuda = cuda
    v1 = jt.array(v_np); v2 = jt.array(v_np)
    idx = jt.array(i_np)
    m1 = group_mean_scatter(v1, idx, G)
    m2 = group_mean_onehot(v2, idx, G)
    fwd = np.abs(m1.numpy() - m2.numpy()).max()
    # 非平凡下游：加权和，梯度非常数
    w = jt.array(rng.randn(G, C).astype(np.float32))
    g1 = jt.grad((m1 * w).sum(), v1).numpy()
    g2 = jt.grad((m2 * w).sum(), v2).numpy()
    bad = np.abs(g1 - g2).max()
    nz1 = (np.abs(g1) > 0).mean(); nz2 = (np.abs(g2) > 0).mean()
    print(f'cuda={cuda} fwd_maxdiff={fwd:.2e} grad_maxdiff={bad:.2e} nonzero: scatter={nz1:.3f} onehot={nz2:.3f}')
