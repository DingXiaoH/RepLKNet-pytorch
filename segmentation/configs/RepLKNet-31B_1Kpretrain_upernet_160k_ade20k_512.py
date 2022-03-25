_base_ = [
    'RepLKNet_upernet.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

#   compared to the default schedule, we used a smaller batchsize/GPU, more GPUs hence fewer training iters
#   please adjust the batchsize and number of iterations according to your own situation
#   we used 8 nodes each with 8 GPUs
#   original default 160k schedule:         160k iters, 4 batchsize per GPU, 8GPUs
#   so with a single node and batchsize=2:  320k iters, 2 batchsize per GPU
#   with 8 nodes and batchsize=2:           40k iters, 2 batchsize per GPU

optimizer = dict(_delete_=True, type='AdamW', lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05, paramwise_cfg=dict(norm_decay_mult=0))

runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=400,             # 1500/160k
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

data=dict(samples_per_gpu=2)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])