_base_ = [
    'RepLKNet_upernet.py',
    '../_base_/datasets/cityscapes_769x769.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

# model settings
model = dict(
    decode_head=dict(
        num_classes=19,
        align_corners=True),
    auxiliary_head=dict(
        num_classes=19,
        align_corners=True),
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