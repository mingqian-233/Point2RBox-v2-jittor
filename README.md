# Point2RBox-v2 (Jittor)

This repository is a [Jittor](https://github.com/Jittor/jittor) port of
**[Point2RBox-v2](https://github.com/VisionXLab/point2rbox-v2)** — point-supervised
oriented object detection via spatial layout among instances (CVPR 2025) — built on top of
[VisionXLab/whollywood-jittor](https://github.com/VisionXLab/whollywood-jittor) (JDet + Jittor).
The original LICENSE and citations are preserved.

Every ported module is backed by **numerical parity tests against the official
PyTorch/mmrotate implementation** (see [Parity testing](#parity-testing)):
same config values, same losses, same LR schedule, matching forward/backward
numerics under fixed weights and inputs.

## What is Point2RBox-v2

Point2RBox-v2 learns **rotated boxes from single-point annotations**. Its core is three
priors on the spatial layout among instances:

1. **Gaussian overlap loss** — instances rarely overlap; modeling boxes as 2D Gaussians
   and penalizing their overlap gives per-instance upper bounds.
2. **Voronoi watershed loss** — watershed on the Voronoi diagram of the points gives
   per-instance lower bounds.
3. **Consistency & edge losses** — a self-supervision branch (rotation/flip/scale views)
   plus edge attraction further refine size and angle, with copy-paste augmentation
   densifying the layout.

Training is either **end-to-end** or **two-stage** (generate pseudo rotated-box labels,
then train a standard rotated FCOS on them; the paper reports the two-stage results).

## Results on DOTA-v1.0

| Setting | Backbone | mAP50 (paper, PyTorch) | mAP50 (this repo, Jittor) | Config | Checkpoint |
|---|---|---|---|---|---|
| End-to-end 1x | R50-FPN* | 41.68 | 48.95 | [cfg](configs/point2rbox_v2/point2rbox_v2_final_fixed.py) | [HF release](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor) |
| Two-stage 1x (pseudo → FCOS) | R50-FPN | 62.61 | **59.39** | [cfg](configs/point2rbox_v2/rotated_fcos_1x_dota_using_pseudo.py) | [HF release](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor) |

*The end-to-end model uses the P3 level only (`strides=[8]`), per the official config.
The reported Jittor numbers are DOTA-v1.0 test-server mAP50: 0.4895466803 for
stage 1 and 0.5939490341 for stage 2. The local stage-2 validation mAP50 is
0.6750881105.

## Installation

Tested environment: Python 3.10, **Jittor 1.3.8.5**, CUDA 11.2 (bundled by Jittor),
**g++-10**, **numpy 1.26.4** (hard requirement, see pitfalls).

```shell
# one-shot environment setup (conda env p2r-jittor with all pins + compiler config)
bash scripts/setup_env.sh

# or manually:
conda create -n p2r-jittor python=3.10
conda activate p2r-jittor
pip install jittor==1.3.8.5 numpy==1.26.4 opencv-python pillow pyyaml tqdm shapely
export cc_path=/usr/bin/g++-10       # jittor JIT compiler; g++-11/13 fail with CUDA 11.2
cd Point2RBox-v2-jittor
python setup.py develop
```

> ⚠️ **Two pins that will silently break things if ignored**
> - `numpy>=2` + jittor 1.3.8.5 produces silent numerical garbage — pin `numpy==1.26.4`.
> - The Jittor JIT compiler must be g++-10 on CUDA 11.2 (`cc_path=/usr/bin/g++-10`).
>
> More pitfalls (with root causes) in [docs/porting_notes.md](docs/porting_notes.md) and
> [docs/environment.md](docs/environment.md).

## Data preparation

Uses standard mmrotate-style split DOTA (`split_ss_dota`: 1024×1024 patches, 200 overlap,
`images/` + `annfiles/` DOTA-txt). Point annotations are derived on the fly from rotated-box
annfiles by `ConvertWeakSupervision` semantics (`point_dummy=1`), identical to the official
pipeline. Set the paths in the config:

```python
train=dict(
    dataset=dict(
        type='P2RV2DOTADataset',
        images_dir='<data_root>/split_ss_dota/trainval/images/',
        annfiles_dir='<data_root>/split_ss_dota/trainval/annfiles/',
        ...))
```

Additional DOTA-txt-format datasets are registered in
`python/jdet/data/mm_datasets.py` (DOTA-v1.5/v2.0, STAR, RSAR, OCD-PCB) with class
tables verbatim from mmrotate `METAINFO`.

## Usage

### End-to-end training / evaluation

```shell
python tools/run_net.py --config-file=configs/point2rbox_v2/point2rbox_v2_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/point2rbox_v2/point2rbox_v2_1x_dota.py --task=test
```

### Two-stage (pseudo-label) pipeline

```shell
# 1. train the end-to-end model (above), then generate pseudo rotated-box labels
#    for the trainval split with the pseudo-generator config (test pipeline on trainval):
python tools/generate_pseudo_labels.py \
    --config=configs/point2rbox_v2/point2rbox_v2_pseudo_generator_dota.py \
    --ckpt=work_dirs/point2rbox_v2_1x_dota/checkpoints/ckpt_12.pkl \
    --out=data/point2rbox_v2_pseudo_labels

# optional parity report against an official-mmrotate pseudo-label export
python tools/compare_pseudo_labels.py \
    /path/to/mmrotate/point2rbox_v2_pseudo_labels.bbox.json \
    data/point2rbox_v2_pseudo_labels.bbox.json

# 2. train rotated FCOS on the pseudo labels:
python tools/run_net.py --config-file=configs/point2rbox_v2/rotated_fcos_1x_dota_using_pseudo.py --task=train
```

The pseudo-label json is field-for-field identical to mmrotate `DOTAMetric.results2json`
output (`{image_id, bbox[cx,cy,w,h,a], score, category_id}`), so artifacts are
interchangeable with the PyTorch pipeline in both directions.

### Converting checkpoints from PyTorch

```shell
python tools/convert_torch_ckpt.py --src <mmrotate .pth> --dst <jdet .pkl>
```

## Parity testing

`tests/parity/` pins the port to the official implementation at four levels
(golden files generated by `tools/dump_golden_*.py` from the PyTorch side are
committed to the repo, so the suite runs without a torch environment):

| Level | Scope | Bar (achieved) |
|---|---|---|
| L0 | all three configs, value-for-value vs mmengine dumps | exact |
| L1 | ops: 2×2 eigh/solve, rotated IoU, NMS, LR schedule (1440 points) | ≤1e-5 |
| L2 | losses & heads fwd+bwd under fixed weights (v2 head, FCOS head, GWD, …) | loss rel ≤1e-3, grads ≤1e-4 (CPU tight) |
| L3 | whole-model forward with real trained weights | rel L2 ≤1.2e-3 |

```shell
python -m pytest tests/parity tests/smoke -q
```

Notable pitfalls found while porting (full list in
[docs/porting_notes.md](docs/porting_notes.md)):

- `Var.stop_grad()` is **in-place** (not torch `detach()`);
- Python loops over tensors build O(N) graph nodes → minutes-long `grad`;
- shoelace polygon area must be computed in centered coords or FMA fusion
  fabricates phantom areas (affects differentiable rotated IoU);
- Jittor's builtin GroupNorm uses one-pass variance `E[x²]−E[x]²` (replaced);
- goldens dumped on CUDA torch must disable TF32.

## Project docs

- [docs/STATUS.md](docs/STATUS.md) — one-page project map (state, artifacts, next steps)
- [docs/PROGRESS.md](docs/PROGRESS.md) — append-only work log
- [docs/config_parity.md](docs/config_parity.md) — config-level alignment notes,
  including the 12 counter-intuitive official settings that are preserved on purpose
- [docs/port_scope.md](docs/port_scope.md) — what is ported, what is pending, what is excluded

## Citation

```bibtex
@inproceedings{yu2025point2rbox2,
  title={Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances},
  author={Yi Yu and Botao Ren and Peiyuan Zhang and Mingxin Liu and Junwei Luo and Shaofeng Zhang and Feipeng Da and Junchi Yan and Xue Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}

@article{yu2025whollywood,
  title={Wholly-WOOD: Wholly Leveraging Diversified-quality Labels for Weakly-supervised Oriented Object Detection},
  author={Yi Yu and Xue Yang and Yansheng Li and Zhenjun Han and Feipeng Da and Junchi Yan},
  year={2025},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
}
```

---

# Wholly-WOOD (upstream base repository)

<details>
<summary>Original Wholly-WOOD README (this fork's base)</summary>

## Introduction
We develop **Wholly-WOOD** (**Wholly** Leveraging Diversified-quality Labels for **W**eakly-supervised **O**riented **O**bject **D**etection), a weakly-supervised OOD framework, capable of wholly leveraging various labeling forms (Points, HBoxes, RBoxes, and their combination) in a unified fashion. By only using HBox for training, our Wholly-WOOD achieves performance very close to that of the RBox-trained counterpart on remote sensing and other areas, which significantly reduces the tedious efforts on labor-intensive annotation for oriented objects.

This project is the [Jittor](https://github.com/Jittor/jittor) implementation of Wholly-WOOD. The code works with **Jittor 1.3.8.5**. It is modified from [JDet](https://github.com/Jittor/JDet), which is an object detection benchmark mainly focus on oriented object detection. PyTorch version: [Wholly-WOOD (PyTorch)](https://github.com/yuyi1005/whollywood).

## Models
This repository contains the Wholly-WOOD model and our series of work on weakly-supervised OOD (i.e. H2RBox, H2RBox-v2, and Point2RBox).

### 1. Wholly-WOOD
```shell
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### 2. H2RBox
```shell
python tools/run_net.py --config-file=configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/h2rbox_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### 3. H2RBox-v2
```shell
python tools/run_net.py --config-file=configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/h2rbox_v2p_obb_r50_adamw_fpn_1x_dota.py --task=test
```

### 4. Point2RBox
```shell
python tools/run_net.py --config-file=configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py --task=train
python tools/run_net.py --config-file=configs/whollywood/point2rbox_obb_r50_adamw_fpn_1x_dota.py --task=test
```

## Datasets
The following datasets are supported in JDet, please check the corresponding document before use.

DOTA1.0/DOTA1.5/DOTA2.0 Dataset: [dota.md](docs/dota.md).

FAIR Dataset: [fair.md](docs/fair.md)

SSDD/SSDD+: [ssdd.md](docs/ssdd.md)

You can also build your own dataset by convert your datas to DOTA format.

## Visualization
You can test and visualize results on your own image sets by:
```shell
python tools/run_net.py --config-file=configs/whollywood/whollywood_obb_r50_adamw_fpn_1x_dota.py --task=vis_test
```
You can choose the visualization style you prefer, for more details about visualization, please refer to [visualization.md](docs/visualization.md).

</details>
