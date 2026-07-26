# Config Parity：官方 mmrotate config ↔ Jittor config 逐项对照

> 真相来源：`/root/ref/Point2RBox-v3/configs/point2rbox_v2/*.py`（解析展平后的 golden
> 在 `tests/parity/golden/config_*.json`，由 `tools/dump_config.py` 生成）。
> 可执行版对照：`tests/parity/test_L0_config.py::TestJittorConfigParity`（10 tests）。
> 本文只记录**映射规则**与**疑似异常清单**，数值一律以 golden 为准。

## 类名 / 写法映射表（铁律一允许变的三类）

| 官方（mmrotate/mmdet/mmengine） | Jittor（JDet） | 说明 |
|---|---|---|
| `mmdet.ResNet, out_indices=(1,2,3)` | `Resnet50, return_stages=['layer2','layer3','layer4']` | 数值语义相同 |
| `init_cfg=Pretrained(torchvision://resnet50)` | `pretrained=True` | 同一份 torchvision 权重 |
| `mmdet.FPN` | `FPN` | 参数逐值相同 |
| `mmdet.FocalLoss` | `FocalLoss` | |
| `GWDLoss`（v3 注册名）/`GDLoss(gwd)` | `GDLoss(loss_type='gwd')` | |
| `optim_wrapper.optimizer=AdamW(...)` + `clip_grad` | `optimizer=AdamW(..., grad_clip=...)` | clip 数值不变 |
| `param_scheduler=[LinearLR, MultiStepLR]` | `scheduler=LinearWarmupMultiStepLR` | LR 序列逐点相等（L1 测试 1440 点 rtol 1e-9） |
| `custom_hooks=[mmdet.SetEpochInfoHook]` | Runner 内建 `model.set_epoch(epoch)` | C3 |
| `train_cfg=EpochBasedTrainLoop(max_epochs=12, val_interval=12)` | `max_epoch=12, eval_interval=12` | |
| `default_hooks.logger interval=50 / checkpoint interval=1` | `log_interval=50 / checkpoint_interval=1` | |
| pipeline `LoadImageFromFile→LoadAnnotations(qbox)→ConvertBoxType→ConvertWeakSupervision` | `P2RV2DOTADataset`（内建，直读 split txt） | qbox→rbox 用 cv2.minAreaRect 原样；point_dummy=1 官方默认 |
| pipeline `Resize((1024,1024), keep_ratio=True)` | `MMRotateResize(min_size=1024, max_size=1024)` | 不复用底座 RotatedResize（会重规范角度） |
| pipeline `RandomFlip(0.75, [h,v,diag])` | `MMRotateRandomFlip(同参数)` | 不复用底座 RotatedRandomFlip（-1 偏移+角度公式不同） |
| `data_preprocessor(mean/std/bgr_to_rgb/pad_size_divisor)` | transforms 的 `Normalize`+`Pad(32)`；mean/std 同时传 model 供 TED 反归一化 | |

## 疑似异常但已确认照抄（铁律二，勿"修正"）

| # | 项 | 已照抄的值 | 状态 |
|---|---|---|---|
| 1 | `_delete_=True` 后 clip_grad 仍生效 | `grad_clip(max_norm=35, norm_type=2)` | ✅ C1 + L0 锚定测试 |
| 2 | param_scheduler 未被覆盖 | LinearLR(1/3, iter 0→500) × MultiStepLR([8,11], 0.1) | ✅ C2，LR 逐点相等 |
| 3 | SetEpochInfoHook 是功能性依赖 | runner 每 epoch 注入 | ✅ C3 |
| 4 | val 指向 trainval/ | 照抄 | ✅ config + L0 测试断言 |
| 5 | stage-2 wd=0.005（≠端到端 0.05） | 照抄 | ✅ golden 锚定（stage-2 config M7 时写） |
| 6 | head strides=[8] 单层 | 照抄 | ✅ |
| 7 | backbone out_indices v2=(1,2,3) / stage-2=(0,1,2,3) | 各是各的 | ✅ |
| 8 | square_cls/edge_loss_cls/post_process/voronoi_thres 逐类魔数 | 一个数不动 | ✅ L0 锚定 |
| 9 | ss_prob=[0.68,0.07,0.25] | 照抄 | ✅ |
| 10 | RandomFlip prob=0.75 | 照抄 | ✅ |
| 11 | boxtype2tensor=False / relu_before_extra_convs=True / add_extra_convs='on_output' / filter_empty_gt=True | 全部显式 | ✅ |
| 12 | `label_assign_pseudo_label_switch_eopch` 拼写 | v3 用，v2 未涉及；若移植中遇到保持原拼写 | 备案 |

## 已知实现层差异（非 config 数值，均有记录与理由）

| 项 | 差异 | 影响 |
|---|---|---|
| torchvision `resized_crop` 的 antialias | jt.nn.interpolate 无 antialias | 仅 sca 增广路径图像内容，训练随机性范围内 |
| mmcv RoIAlignRotated aligned=True 不做 1x1 钳制 | JDet kernel 始终钳制 | 仅亚像素 roi |
| jdet nms keep 顺序 | head 内先按分数排序再 NMS，语义对齐 mmcv | 已适配 |
| reduce_mean（分布式均值） | 单卡训练恒等 | 官方总 batch=2 单卡，无差异 |
| eigh 简并子梯度基底 | w==h 时与 torch 的 a/c 分配不同（trace 一致） | 数学等价的子梯度选择 |
