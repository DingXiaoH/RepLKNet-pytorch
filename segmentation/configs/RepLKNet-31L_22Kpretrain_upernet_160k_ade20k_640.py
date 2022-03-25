_base_ = ['RepLKNet-31B_22Kpretrain_upernet_160k_ade20k_640.py',]

model = dict(
    backbone=dict(
        channels=[192, 384, 768, 1536],
        drop_path_rate=0.3,
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536]
    ),
    auxiliary_head=dict(
        in_channels=768
    ),
)
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=800,             # 1500/160k
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

#   compared to the default schedule, we used a smaller batchsize/GPU, more GPUs hence fewer training iters
#   please adjust the batchsize and number of iterations according to your own situation
#   we used 8 nodes each with 8 GPUs
#   original default 160k schedule:         160k iters, 4 batchsize per GPU, 8GPUs
#   so with a single node and batchsize=1:  640k iters, 1 batchsize per GPU
#   with 8 nodes and batchsize=1:           80k iters, 1 batchsize per GPU

data=dict(samples_per_gpu=1)

runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')