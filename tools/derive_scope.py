"""derive_scope.py — 机械化推导 Point2RBox-v2 三个官方 config 的移植闭包（铁律三）。

在 p2r-torch 环境运行：
    cd /root/ref/Point2RBox-v3 && PYTHONPATH=/root/ref/Point2RBox-v3 \
        python /root/work/A/Point2RBox-v2-jittor/tools/derive_scope.py

步骤：
  1) Config.fromfile 解析 3 个 config，递归收集所有 type='...' 字符串
  2) 经 mmengine registry 把每个 type 解析到定义它的 .py 文件
  3) 从这些文件出发做 import 传递闭包（只追 mmrotate.* / third_parties.*；
     __init__.py 不展开——注册链按铁律三第 2 层"全部包含"处理，展开只会把全仓库拉进来）
  4) 三个 config 取并集，按来源分类输出 markdown 到 stdout
"""
import ast
import inspect
import os
import sys
from collections import defaultdict

REPO = os.environ.get('P2R_REF', '/root/ref/Point2RBox-v3')
CONFIGS = [
    'configs/point2rbox_v2/point2rbox_v2-1x-dota.py',
    'configs/point2rbox_v2/point2rbox_v2-pseudo-generator-dota.py',
    'configs/point2rbox_v2/rotated-fcos-1x-dota-using-pseudo.py',
]


def collect_types(node, out):
    if isinstance(node, dict):
        t = node.get('type')
        if isinstance(t, str):
            out.add(t)
        for v in node.values():
            collect_types(v, out)
    elif isinstance(node, (list, tuple)):
        for v in node:
            collect_types(v, out)


def resolve_type(name, registries):
    """type 字符串 → (定义类的 .py 文件, registry 名)；解析不到返回 (None, None)。"""
    for rname, reg in registries:
        try:
            cls = reg.get(name)
        except Exception:
            cls = None
        if cls is not None:
            try:
                return inspect.getfile(cls), rname
            except TypeError:
                return None, rname
    return None, None


def module_to_file(mod):
    """'mmrotate.a.b' → 仓库内文件路径（.py 或包的 __init__.py）；不存在返回 None。"""
    rel = mod.replace('.', '/')
    for cand in (os.path.join(REPO, rel + '.py'),
                 os.path.join(REPO, rel, '__init__.py')):
        if os.path.isfile(cand):
            return cand
    return None


def file_to_module(path):
    rel = os.path.relpath(path, REPO)
    if rel.endswith('/__init__.py'):
        rel = rel[:-len('/__init__.py')]
    elif rel.endswith('.py'):
        rel = rel[:-3]
    return rel.replace('/', '.')


def imports_of(pyfile):
    """解析文件的 import，返回 mmrotate.* / third_parties.* 的目标文件集合。"""
    with open(pyfile) as f:
        tree = ast.parse(f.read(), pyfile)
    pkg = file_to_module(pyfile)
    if not pyfile.endswith('__init__.py'):
        pkg = pkg.rsplit('.', 1)[0] if '.' in pkg else ''
    targets = set()
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # 相对 import
                base = pkg.rsplit('.', node.level - 1)[0] if node.level > 1 else pkg
                mod = base + ('.' + node.module if node.module else '')
            else:
                mod = node.module or ''
            mods = [mod]
            # from X import Y 里 Y 可能是子模块
            mods += [f'{mod}.{a.name}' for a in node.names]
        for m in mods:
            if m.startswith(('mmrotate', 'third_parties')):
                f2 = module_to_file(m)
                if f2:
                    targets.add(f2)
    return targets


def main():
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    from mmengine.config import Config
    from mmrotate.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    # 注意：Registry.get 只向父链搜索。必须用 mmrotate 的子 registry（父链可达
    # mmdet/mmengine），用 mmengine 根 registry 会看不到 mmrotate 的注册项。
    from mmrotate import registry as RR
    from mmengine import registry as R
    names = ('MODELS', 'DATASETS', 'TRANSFORMS', 'TASK_UTILS', 'METRICS', 'HOOKS',
             'DATA_SAMPLERS', 'OPTIMIZERS', 'OPTIM_WRAPPERS', 'PARAM_SCHEDULERS',
             'LOOPS', 'EVALUATOR', 'VISUALIZERS', 'VISBACKENDS', 'LOG_PROCESSORS',
             'RUNNERS', 'MODEL_WRAPPERS', 'WEIGHT_INITIALIZERS')
    registries = [(n, getattr(RR, n)) for n in names if hasattr(RR, n)]
    registries += [(n, getattr(R, n)) for n in names if not hasattr(RR, n)]

    per_cfg_types = {}
    all_types = set()
    for c in CONFIGS:
        cfg = Config.fromfile(os.path.join(REPO, c))
        ts = set()
        collect_types(cfg.to_dict(), ts)
        per_cfg_types[c] = ts
        all_types |= ts

    seed_files, type_rows, unresolved = set(), [], []
    for t in sorted(all_types):
        f, rname = resolve_type(t, registries)
        used_in = [os.path.basename(c) for c in CONFIGS if t in per_cfg_types[c]]
        if f is None:
            unresolved.append((t, used_in))
            continue
        type_rows.append((t, f, rname, used_in))
        if f.startswith(REPO):
            seed_files.add(f)

    # import 传递闭包（__init__.py 不展开）
    closure, queue = set(seed_files), sorted(seed_files)
    while queue:
        cur = queue.pop()
        if cur.endswith('__init__.py'):
            continue
        for dep in imports_of(cur):
            if dep not in closure:
                closure.add(dep)
                queue.append(dep)

    # ---- 输出 ----
    print('<!-- 本节由 tools/derive_scope.py 自动生成，重跑会覆盖 -->')
    print('\n## A. config 中出现的全部 type 及其解析\n')
    print('| type | registry | 定义位置 | 出现于 |')
    print('|---|---|---|---|')
    for t, f, rname, used in type_rows:
        loc = os.path.relpath(f, REPO) if f.startswith(REPO) else \
            f.split('site-packages/')[-1] if 'site-packages' in f else f
        origin = '**mmrotate（需移植）**' if f.startswith(REPO) else '框架（JDet core 对应）'
        print(f'| `{t}` | {rname} | `{loc}`<br>{origin} | {", ".join(used)} |')
    for t, used in unresolved:
        print(f'| `{t}` | ❓未解析 | — | {", ".join(used)} |')

    print('\n## B. mmrotate/third_parties 侧 import 传递闭包（去 __init__，共 %d 个文件）\n'
          % len(closure))
    by_dir = defaultdict(list)
    for f in sorted(closure):
        rel = os.path.relpath(f, REPO)
        by_dir[os.path.dirname(rel)].append(rel)
    for d in sorted(by_dir):
        print(f'- **{d}/**')
        for f in by_dir[d]:
            star = ' ← seed(registry 直达)' if os.path.join(REPO, f) in seed_files else ''
            print(f'  - `{os.path.basename(f)}`{star}')

    print('\n<!-- 生成结束 -->')


if __name__ == '__main__':
    main()
