# Demo
To run the demo (test) we need to run the **demo/inference.py** file wiht three main parameters as follows:

```bash
python demo/inference.py [path to config file] [path to trained model] --images [path to image to test]
```
a correct example:

```bash
python demo/inference.py local_configs/segformer/B2/segformer.b2.512x512.sber.160k.py work_dirs/segformer.b2.512x512.sber.160k/iter_160000.pth --images /home/ghadeer/Projects/Datasets/RICOH_THETA_resized/
```

The results will be created in the file ```results``` inside a file named as the current time (the last number file is the newest results) then it can be changed manually to the correct name *(the used model for example)* 
______________________
### config files:
The config files are in the **local_configs** file.
For each model (B0, B1, ..., B5) there are different config_files.
for each dataset we need to add a new config file:
for example, if we need to use the model B2 (512*512) for Sberbank dataset, we need to **copy** an original config file with the same characteristics and modify it for the new dataset **for example:** **segformer.b2.512x512.ade.160k.py**
______________________
#### How to change the config file
There are multiple things to change in the config file:
the original config file can be as follows:

```python
_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b2.pth',
    backbone=dict(
        type='mit_b2',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        # type='MLPHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

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

data = dict(samples_per_gpu=2)
evaluation = dict(interval=16000, metric='mIoU')
```

We need to change the second line in the **_base_** parameter and replace the **ade20k** dataset file with **Sber** dataset file, so it becomes 

```python
_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/sber_512x512_repeat.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]
```

Also we need to change the number of classes in the **model** in **decode_head** (change it from 150 to 6 **"in the case of sberbank dataset there are 6 classes"**)

```python
        # num_classes=150,
        num_classes=6,
```

### Change the model
if we need to change the structure of the neural network we also need to change it in the config file:
for example to use the structure for neural network with edges we need to change the type of the **decode_head** from ```SegFormerHead``` to ```SegFormerheadWithEdges```
```python
decode_head=dict(
    # type='EncoderDecoder',
    type='SegFormerheadWithEdges',
```

the structure ```SegFormerheadWithEdges``` was designed and added to ```/SegFormer/mmseg/models/decode_heads/segformer_head_with_edges.py```

#### Create new model
To create a new decoder it is necessary to create the new decoder in ```/SegFormer/mmseg/models/decode_heads/``` 
if the new decoder is not very different from the Segformer decoder it is possible to copy ```segformer_head.py``` and edit it, **But** in the case of adding edges, it was not possible because I needed to add new loss function, so I needed to copy and edit the original code of the decoders ```decode_head.py``` 
__________________________
### Change the dataset:
To use new dataset we need to create a config file for the new dataset in ```local_configs/_base_/datasets```
for reference it is possible to compare with the config files created for Sberbank dataset **sber_512x512_repeat.py** & **sber_repeat.py**
the crop size in the dataset should be the same as the crop size in the model.
For example in the ```sber_512x512_repeat.py``` config file, we use the 512*512 cropping size, and in ```segformer.b2.512x512.sber.160k.py``` we also use the same cropping size

The code inside the ```sber_512x512_repeat.py```
```python
# dataset settings
dataset_type = 'SberbankDataset'
data_root = 'data/SberMerged/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
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
```
The important parts to change are (pay attention to):
- The dataset type
```python
dataset_type = 'SberbankDataset'
```
This type should be created and added to ```mmseg/datasets/__init__.py```.
For more detailed information please see the files ```mmseg/datasets/__init__.py``` and ```mmseg/datasets/sber.py```
- The root to the dataset (Here I copied the dataset inside the folder **data**)
```python
data_root = 'data/SberMerged/'
```
- The crop size (to match the used model)
```python
crop_size = (512, 512)
```
- The paths to the RGB image and GT mask
```python
    train=dict(
            .  
            .
        img_dir='train/images',
        ann_dir='train/Semantic_palette',
            .
            .

    val=dict(
            .  
            .
        img_dir='validation/images',
        ann_dir='validation/Semantic_palette',
            .
            .
    test=dict(
            .  
            .
        img_dir='test/images',
        ann_dir='test/Semantic_palette',
```
_______
# Train
to train a SegFormer model you need to run the following code:
```bash
./tools/dist_train.sh [path to config file] [number of GPUs to use]
```
for example

```bash
./tools/dist_train.sh local_configs/segformer/B5/segformer.b5.512x512.sber.160k.py 2
```
in the local_configs file we define the dataset and the training parameters

The trained models will be saved in ```work_dirs``` inside a file named the same name as the used model.