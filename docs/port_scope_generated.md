<!-- 本节由 tools/derive_scope.py 自动生成，重跑会覆盖 -->

## A. config 中出现的全部 type 及其解析

| type | registry | 定义位置 | 出现于 |
|---|---|---|---|
| `AdamW` | OPTIMIZERS | `torch/optim/adamw.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `BN` | MODELS | `torch/nn/modules/batchnorm.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `CheckpointHook` | HOOKS | `mmengine/hooks/checkpoint_hook.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `ConvertBoxType` | TRANSFORMS | `mmrotate/datasets/transforms/transforms.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `ConvertWeakSupervision` | TRANSFORMS | `mmrotate/datasets/transforms/transforms.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `DOTADataset` | DATASETS | `mmrotate/datasets/dota.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `DOTAMetric` | METRICS | `mmrotate/evaluation/metrics/dota_metric.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `DefaultSampler` | DATA_SAMPLERS | `mmengine/dataset/sampler.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `DistSamplerSeedHook` | HOOKS | `mmengine/hooks/sampler_seed_hook.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `DistanceAnglePointCoder` | TASK_UTILS | `mmrotate/models/task_modules/coders/distance_angle_point_coder.py`<br>**mmrotate（需移植）** | rotated-fcos-1x-dota-using-pseudo.py |
| `EdgeLoss` | MODELS | `mmrotate/models/losses/point2rbox_v2_loss.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `EpochBasedTrainLoop` | LOOPS | `mmengine/runner/loops.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `GDLoss` | MODELS | `mmrotate/models/losses/gaussian_dist_loss.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `GaussianOverlapLoss` | MODELS | `mmrotate/models/losses/point2rbox_v2_loss.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `IterTimerHook` | HOOKS | `mmengine/hooks/iter_timer_hook.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `LinearLR` | PARAM_SCHEDULERS | `mmengine/optim/scheduler/lr_scheduler.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `LocalVisBackend` | VISBACKENDS | `mmengine/visualization/vis_backend.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `LogProcessor` | LOG_PROCESSORS | `mmengine/runner/log_processor.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `LoggerHook` | HOOKS | `mmengine/hooks/logger_hook.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `MultiStepLR` | PARAM_SCHEDULERS | `mmengine/optim/scheduler/lr_scheduler.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `OptimWrapper` | OPTIM_WRAPPERS | `mmengine/optim/optimizer/optimizer_wrapper.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `PSCCoder` | TASK_UTILS | `mmrotate/models/task_modules/coders/angle_coder.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `ParamSchedulerHook` | HOOKS | `mmengine/hooks/param_scheduler_hook.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `Point2RBoxV2` | MODELS | `mmrotate/models/detectors/point2rbox_v2.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `Point2RBoxV2ConsistencyLoss` | MODELS | `mmrotate/models/losses/point2rbox_v2_loss.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `Point2RBoxV2Head` | MODELS | `mmrotate/models/dense_heads/point2rbox_v2_head.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `Pretrained` | WEIGHT_INITIALIZERS | `mmengine/model/weight_init.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `RotLocalVisualizer` | VISUALIZERS | `mmrotate/visualization/local_visualizer.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `RotatedFCOSHead` | MODELS | `mmrotate/models/dense_heads/rotated_fcos_head.py`<br>**mmrotate（需移植）** | rotated-fcos-1x-dota-using-pseudo.py |
| `RotatedIoULoss` | MODELS | `mmrotate/models/losses/rotated_iou_loss.py`<br>**mmrotate（需移植）** | rotated-fcos-1x-dota-using-pseudo.py |
| `TestLoop` | LOOPS | `mmengine/runner/loops.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `ValLoop` | LOOPS | `mmengine/runner/loops.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `VoronoiWatershedLoss` | MODELS | `mmrotate/models/losses/point2rbox_v2_loss.py`<br>**mmrotate（需移植）** | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `mmdet.CrossEntropyLoss` | MODELS | `mmdet/models/losses/cross_entropy_loss.py`<br>框架（JDet core 对应） | rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.DetDataPreprocessor` | MODELS | `mmdet/models/data_preprocessors/data_preprocessor.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.DetVisualizationHook` | HOOKS | `mmdet/engine/hooks/visualization_hook.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.FCOS` | MODELS | `mmdet/models/detectors/fcos.py`<br>框架（JDet core 对应） | rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.FPN` | MODELS | `mmdet/models/necks/fpn.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.FocalLoss` | MODELS | `mmdet/models/losses/focal_loss.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.LoadAnnotations` | TRANSFORMS | `mmdet/datasets/transforms/loading.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.LoadImageFromFile` | TRANSFORMS | `mmcv/transforms/loading.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.PackDetInputs` | TRANSFORMS | `mmdet/datasets/transforms/formatting.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.RandomFlip` | TRANSFORMS | `mmdet/datasets/transforms/transforms.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.ResNet` | MODELS | `mmdet/models/backbones/resnet.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.Resize` | TRANSFORMS | `mmdet/datasets/transforms/transforms.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |
| `mmdet.SetEpochInfoHook` | HOOKS | `mmdet/engine/hooks/set_epoch_info_hook.py`<br>框架（JDet core 对应） | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py |
| `nms_rotated` | ❓未解析 | — | point2rbox_v2-1x-dota.py, point2rbox_v2-pseudo-generator-dota.py, rotated-fcos-1x-dota-using-pseudo.py |

## B. mmrotate/third_parties 侧 import 传递闭包（去 __init__，共 22 个文件）

- **mmrotate/**
  - `registry.py`
- **mmrotate/datasets/**
  - `__init__.py`
  - `dota.py` ← seed(registry 直达)
- **mmrotate/datasets/transforms/**
  - `transforms.py` ← seed(registry 直达)
- **mmrotate/evaluation/**
  - `__init__.py`
- **mmrotate/evaluation/metrics/**
  - `dota_metric.py` ← seed(registry 直达)
- **mmrotate/models/dense_heads/**
  - `point2rbox_v2_head.py` ← seed(registry 直达)
  - `rotated_fcos_head.py` ← seed(registry 直达)
- **mmrotate/models/detectors/**
  - `point2rbox_v2.py` ← seed(registry 直达)
- **mmrotate/models/losses/**
  - `gaussian_dist_loss.py` ← seed(registry 直达)
  - `point2rbox_v2_loss.py` ← seed(registry 直达)
  - `rotated_iou_loss.py` ← seed(registry 直达)
  - `utils.py`
  - `vis.py`
- **mmrotate/models/task_modules/coders/**
  - `angle_coder.py` ← seed(registry 直达)
  - `distance_angle_point_coder.py` ← seed(registry 直达)
- **mmrotate/structures/**
  - `__init__.py`
- **mmrotate/structures/bbox/**
  - `__init__.py`
  - `transforms.py`
- **mmrotate/visualization/**
  - `local_visualizer.py` ← seed(registry 直达)
  - `palette.py`
- **third_parties/ted/**
  - `ted.py`

<!-- 生成结束 -->
