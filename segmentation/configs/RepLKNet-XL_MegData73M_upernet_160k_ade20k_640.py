_base_ = ['RepLKNet-31L_22Kpretrain_upernet_160k_ade20k_640.py']

model = dict(
    backbone=dict(
        large_kernel_sizes=[27,27,27,13],
        channels=[256, 512, 1024, 2048],
        drop_path_rate=0.5,
        small_kernel=None,
        dw_ratio=1.5,
        norm_intermediate_features=True
    ),
    decode_head=dict(
        in_channels=[256, 512, 1024, 2048]
    ),
    auxiliary_head=dict(
        in_channels=1024
    ),
)

#   compared to the default schedule, we used a smaller batchsize/GPU, more GPUs hence fewer training iters
#   please adjust the batchsize and number of iterations according to your own situation
#   we used 2 nodes (A100) each with 8 GPUs
#   original default 160k schedule:         160k iters, 4 batchsize per GPU, 8GPUs
#   so with a single node and batchsize=2:  320k iters, 2 batchsize per GPU
#   with 2 nodes and batchsize=2:           160k iters, 2 batchsize per GPU

crop_size = (640, 640)

#Note this. The mean/std should agree with the pretraining
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)

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
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    train=dict(pipeline=test_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)