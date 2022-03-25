_base_ = [
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='RepLKNet',
        large_kernel_sizes=[31,29,27,13],
        layers=[2,2,18,2],
        channels=[128,256,512,1024],
        drop_path_rate=0.5,
        small_kernel=5,
        dw_ratio=1,
        num_classes=None,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=True,
        small_kernel_merged=False,
        use_sync_bn=True        #   Note: use SyncBN
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05, paramwise_cfg=dict(norm_decay_mult=0))

#   original (default) single-node setting: 8GPUs, 80k iters
#   we used 4 nodes each with 8 GPUs:       32GPUs, 20k iters

runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=400,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])