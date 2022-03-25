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
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
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
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))