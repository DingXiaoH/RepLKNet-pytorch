_base_ = [
    'RepLKNet_upernet.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py',
]

crop_size = (640, 640)

model = dict(
    backbone=dict(
        drop_path_rate=0.3,
    ),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(426, 426))
)

#   increase training crop_size from 512 to 640.
#   the batchsize is 2, same as the 512x512 training setting
#   other settings unchanged

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
data = dict(
    samples_per_gpu=2,
    train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05, paramwise_cfg=dict(norm_decay_mult=0))

#   compared to the default schedule, we used a smaller batchsize/GPU, more GPUs hence fewer training iters
#   please adjust the batchsize and number of iterations according to your own situation
#   we used 8 nodes each with 8 GPUs
#   original default 160k schedule:         160k iters, 4 batchsize per GPU, 8GPUs
#   so with a single node and batchsize=2:  320k iters, 2 batchsize per GPU
#   with 8 nodes and batchsize=2:           40k iters, 2 batchsize per GPU

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=400,             # 1500/160k
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
