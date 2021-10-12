# dataset settings
dataset_type = 'SberbankDatasetWithDepth'
data_root = 'data/SberMerged_RGBD/'
# data_root = 'data/SberMerged/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

depth_norm_cfg_train = dict(
    mean=[3896.0], std=[4984.3], data_name="depth")
depth_norm_cfg_test = dict(
    mean=[3978.6], std=[5048.8], data_name="depth")
# img_norm_cfg = dict(
#     mean=[0., 0., 0.], std=[1., 1., 1.], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile', depth_type='repeat', depth_channels='grayscale'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0), apply_to_channels=['depth']),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='NormalizeData', **depth_norm_cfg_train),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='WithDepthFormatBundle'),                             # Transforme arrays to tensors with Depth 
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], additional_meta_keys=['channels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthFromFile', depth_type='repeat', depth_channels='grayscale'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='NormalizeData', **depth_norm_cfg_test),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], additional_meta_keys=['channels']),
        ])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=500,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/images',
            ann_dir='train/Semantic_palette',
            # depth_dir='train/depth_png',
            # depth_suffix='.png',
            depth_dir='train/normalized_depth',
            depth_suffix='.npy',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='validation/images',
        ann_dir='validation/Semantic_palette',
        # depth_dir='validation/depth_png',
        # depth_suffix='.png',
        depth_dir='validation/normalized_depth',
        depth_suffix='.npy',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        ann_dir='test/Semantic_palette',
        # depth_dir='test/depth_png',
        # depth_suffix='.png',
        depth_dir='test/normalized_depth',
        depth_suffix='.npy',
        pipeline=test_pipeline))
