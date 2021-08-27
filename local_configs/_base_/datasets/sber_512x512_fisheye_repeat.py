# dataset settings
dataset_type = 'SberbankDatasetFisheye'
data_root = 'data/SberMerged/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadFisheyeImageFromFile', bbox=[[71, 124], [1370, 1316]]),
    dict(type='LoadAnnotations'),
    dict(type='RandomFisheyeShift', prob=0.5, max_dx=200, max_dy=200, palette=True),
    dict(type='RandomFisheyeCrop', prob=0.5,
         part_x_range=(0.8, 1.2), part_y_range=(0.8, 1.2), palette=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Resize', img_scale=crop_size, keep_ratio=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadFisheyeImageFromFile', bbox=[[71, 124], [1370, 1316]]),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/images_fisheye',
            ann_dir='train/Semantic_fisheye_palette',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='validation/images_fisheye',
        ann_dir='validation/Semantic_fisheye_palette',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images_fisheye',
        ann_dir='test/Semantic_fisheye_palette',
        pipeline=test_pipeline))
