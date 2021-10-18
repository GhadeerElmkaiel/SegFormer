_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/rgbd_mirror_512x512_hha.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='GeneralEncoderDecoder',
    pretrained='pretrained/mit_b2.pth',
    backbone=dict(
        type='mit_depth_b2',
        depth_embed_type='HHA',
        weights_only_MVF=True),
    decode_head=dict(
        type='SegFormerheadWithDepthEdges',
        in_channels=[64, 128, 320, 512, 64, 128, 320, 512],
        in_index=[0, 1, 2, 3, 4, 5, 6, 7],
        feature_strides=[4, 8, 16, 32, 4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512,512), stride=(384,384)))

# data
data = dict(samples_per_gpu=1, workers_per_gpu=2)
# checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=4000, metric='mIoU')

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


init_kwargs = dict(config=dict(data=data, model=model, optimizer=optimizer))

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
        dict(type='WandbLoggerHook', by_epoch=False, init_kwargs=init_kwargs)
    ])
