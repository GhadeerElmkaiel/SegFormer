# dataset settings
dataset_type = 'SberbankDatasetFisheye'
data_root = 'data/SberMerged/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='DistortPinholeToFisheye', transform_probability = 1.,  input_shape = (720,1280), output_shape = (1200,1920), num_classes=6,
                                        focal_distances = [25, 50, 75, 100, 125, 150], maps_probability = None, palette=True),
    dict(type='RandomFisheyeShift', prob=0.5, max_dx=200, max_dy=200, num_classes=6, palette=True),
    dict(type='RandomFisheyeCrop', prob=0.5, num_classes=6, max_dx=100, max_dy=100,
         part_x_range=(0.8, 1.4), part_y_range=(0.8, 1.4), palette=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='DistortPinholeToFisheye', transform_probability = 1.,  input_shape = (720,1280), output_shape = (1200,1920), num_classes=6,
                focal_distances = [80], maps_probability = None, palette=True),
            dict(type='RandomFisheyeCrop', prob=1, num_classes=6, dy_range=(30,31),
                part_x_range=(1.29, 1.31), part_y_range=(0.89, 0.89), palette=True),
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
            img_dir='train/images',
            ann_dir='train/Semantic_palette',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='validation/images',
        ann_dir='validation/Semantic_palette',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/Semantic_palette',
        pipeline=test_pipeline))
