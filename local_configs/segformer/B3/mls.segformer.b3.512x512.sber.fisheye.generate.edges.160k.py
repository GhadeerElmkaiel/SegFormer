_base_ = [
    '/home/jovyan/segformer/local_configs/_base_/models/segformer.py',
    '/home/jovyan/segformer/local_configs/_base_/datasets/sber_512x512_fisheye_generate_repeat.py',
    '/home/jovyan/segformer/local_configs/_base_/default_runtime.py',
    '/home/jovyan/segformer/local_configs/_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/home/jovyan/segformer/pretrained/mit_b3.pth',
    backbone=dict(
        type='mit_b5',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerheadWithEdges',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=7,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# data
data = dict(samples_per_gpu=3, workers_per_gpu=3)
checkpoint_config = dict(by_epoch=False, interval=16000)
evaluation = dict(interval=16000, metric='mIoU')

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

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerImagesHook', num_classes=7, img_interval=1000,
                    log_dir='/home/jovyan/segformer/work_dirs/tf_board')
    ])


