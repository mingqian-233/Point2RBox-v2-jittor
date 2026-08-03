# Point2RBox-v2 for Jittor

Jittor/JDet implementation of [Point2RBox-v2](https://github.com/VisionXLab/point2rbox-v2),
a point-supervised oriented object detector based on spatial relationships between
instances. This repository contains the end-to-end model, pseudo-label export, the
rotated-FCOS second stage, DOTA evaluation utilities, checkpoint conversion tools,
and numerical parity tests against the PyTorch reference.

## Results

DOTA-v1.0 Task1, mAP50 on the official test server:

| Model | Paper | Jittor | Checkpoint |
|---|---:|---:|---|
| Point2RBox-v2 end-to-end | 51.00 | 48.95 | [download](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor/resolve/main/stage1/point2rbox_v2_stage1_ckpt_12.pkl) |
| Point2RBox-v2 + rotated-FCOS | 62.61 | **59.39** | [download](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor/resolve/main/stage2/rotated_fcos_stage2_ckpt_12.pkl) |

The corresponding local trainval-patch diagnostics are 54.5264 and 67.5088 mAP50.
These local values are not the official test-server metric. Weights, official-format
Task1 submissions, checksums, and the complete metric record are available in the
[Hugging Face repository](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor).

## Environment

The validated environment uses Python 3.10, Jittor 1.3.8.5, NumPy 1.26.4,
Jittor's bundled CUDA 11.2, and g++-10. NumPy must remain below version 2 with
this Jittor release.

```bash
bash scripts/setup_env.sh
conda activate p2r-jittor
export cc_path=/usr/bin/g++-10
export PYTHONPATH="$PWD:$PWD/python"
```

For an existing Python 3.10 environment, install `requirements.txt`, pin
`jittor==1.3.8.5` and `numpy==1.26.4`, then run `python setup.py develop`.
See [docs/environment.md](docs/environment.md) for compiler and CUDA compatibility
notes.

## Data and weights

Prepare DOTA-v1.0 as 1024×1024 patches with gap 200. The recovered training run used
12,800 trainval patches and 2,709 test patches with this layout:

```text
/path/to/split_ss_dota/
├── trainval/
│   ├── images/
│   └── annfiles/
└── test/
    └── images/
```

Update `images_dir`, `annfiles_dir`, and `ann_json` in the selected config when the
dataset is stored elsewhere. Dataset preparation details are in
[docs/dota.md](docs/dota.md).

Download the final weights from Hugging Face:

```bash
huggingface-cli download Mingqian-233/Point2RBox-v2-jittor \
  stage1/point2rbox_v2_stage1_ckpt_12.pkl \
  stage2/rotated_fcos_stage2_ckpt_12.pkl \
  --local-dir weights
```

## Training

Train the point-supervised end-to-end model:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v2/point2rbox_v2_final_fixed.py \
  --task train
```

Generate rotated-box pseudo labels from the final stage-1 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/generate_pseudo_labels.py \
  --config configs/point2rbox_v2/point2rbox_v2_pseudo_generator_dota.py \
  --ckpt work_dirs/point2rbox_v2_1x_dota_final_fixed/checkpoints/ckpt_12.pkl \
  --out /path/to/split_ss_dota/point2rbox_v2_pseudo_labels
```

Set `ann_json` in the stage-2 config to the generated `.bbox.json`, then train the
rotated-FCOS detector:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v2/rotated_fcos_1x_dota_using_pseudo.py \
  --task train
```

The configs reproduce the reference 12-epoch schedule, including 500-iteration
linear warmup, epoch milestones 8 and 11, and gradient clipping at 35. The completed
unattended stage-1 → pseudo-label → stage-2 workflow is recorded in
[`tools/auto_stage2_pipeline.sh`](tools/auto_stage2_pipeline.sh).

## Evaluation

Set `resume_path` in the selected config to the downloaded checkpoint and run:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v2/rotated_fcos_1x_dota_using_pseudo.py \
  --task val

CUDA_VISIBLE_DEVICES=0 python tools/run_net.py \
  --config-file configs/point2rbox_v2/rotated_fcos_1x_dota_using_pseudo.py \
  --task test
```

Testing writes patch predictions, merges them back to the original DOTA images, and
creates a submission archive under `submit_zips/`. Official submissions must contain
exactly 15 files named `Task1_<class>.txt`; the automated pipeline validates this
layout before publishing.

Ready-to-submit archives are also available directly from Hugging Face:

- [stage-1 Task1 submission](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor/resolve/main/stage1/dota_task1_submission.zip)
- [stage-2 Task1 submission](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor/resolve/main/stage2/dota_task1_submission.zip)

## Verification

The parity and smoke suites cover configuration values, geometry and rotated ops,
losses and gradients, detector routing, checkpoint conversion, pseudo-label
serialization, and dataset adapters.

```bash
export cc_path=/usr/bin/g++-10
export PYTHONPATH="$PWD:$PWD/python"
python -m pytest tests/parity tests/smoke -q
```

The release result is `65 passed, 2 skipped`. Porting details are documented in
[docs/porting_notes.md](docs/porting_notes.md), with exact configuration mappings in
[docs/config_parity.md](docs/config_parity.md).

## Acknowledgements

This implementation is built on [JDet](https://github.com/Jittor/JDet),
[Wholly-WOOD for Jittor](https://github.com/VisionXLab/whollywood-jittor), and the
original [Point2RBox-v2](https://github.com/VisionXLab/point2rbox-v2).

Released under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt).

## Citation

```bibtex
@inproceedings{yu2025point2rbox2,
  title={Point2RBox-v2: Rethinking Point-supervised Oriented Object Detection with Spatial Layout Among Instances},
  author={Yi Yu and Botao Ren and Peiyuan Zhang and Mingxin Liu and Junwei Luo and Shaofeng Zhang and Feipeng Da and Junchi Yan and Xue Yang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
