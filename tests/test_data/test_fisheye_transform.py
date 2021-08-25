import copy
import os.path as osp

import mmcv
import numpy as np
import pytest
from mmcv.utils import build_from_cfg
from PIL import Image

from mmseg.datasets.builder import PIPELINES

PATH_TO_IMAGE = osp.join(osp.dirname(__file__),'..', '..','data', 'SberMerged','test', 'images_fisheye', '1.png')
PATH_TO_SEG = osp.join(osp.dirname(__file__),'..', '..','data', 'SberMerged','test', 'Semantic_fisheye', '1.png')

def test_resize():
    # test assertion if img_scale is a list
    with pytest.raises(AssertionError):
        transform = dict(type='Resize', img_scale=[1333, 800], keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion if len(img_scale) while ratio_range is not None
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            ratio_range=(0.9, 1.1),
            keep_ratio=True)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid multiscale_mode
    with pytest.raises(AssertionError):
        transform = dict(
            type='Resize',
            img_scale=[(1333, 800), (1333, 600)],
            keep_ratio=True,
            multiscale_mode='2333')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)
    resize_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    # (288, 512, 3)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    assert img.shape == (1550, 1960, 3)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (800, 1012, 3)

    # test keep_ratio=False
    transform = dict(
        type='Resize',
        img_scale=(1280, 800),
        multiscale_mode='value',
        keep_ratio=False)
    resize_module = build_from_cfg(transform, PIPELINES)
    resized_results = resize_module(results.copy())
    assert resized_results['img_shape'] == (800, 1280, 3)

    # # test multiscale_mode='range'
    # transform = dict(
    #     type='Resize',
    #     img_scale=[(1333, 400), (1333, 1200)],
    #     multiscale_mode='range',
    #     keep_ratio=True)
    # resize_module = build_from_cfg(transform, PIPELINES)
    # resized_results = resize_module(results.copy())
    # assert max(resized_results['img_shape'][:2]) <= 1333
    # assert min(resized_results['img_shape'][:2]) >= 400
    # assert min(resized_results['img_shape'][:2]) <= 1200

    # # test multiscale_mode='value'
    # transform = dict(
    #     type='Resize',
    #     img_scale=[(1333, 800), (1333, 400)],
    #     multiscale_mode='value',
    #     keep_ratio=True)
    # resize_module = build_from_cfg(transform, PIPELINES)
    # resized_results = resize_module(results.copy())
    # assert resized_results['img_shape'] in [(750, 1333, 3), (400, 711, 3)]

    # # test multiscale_mode='range'
    # transform = dict(
    #     type='Resize',
    #     img_scale=(1333, 800),
    #     ratio_range=(0.9, 1.1),
    #     keep_ratio=True)
    # resize_module = build_from_cfg(transform, PIPELINES)
    # resized_results = resize_module(results.copy())
    # assert max(resized_results['img_shape'][:2]) <= 1333 * 1.1

    # # test img_scale=None and ratio_range is tuple.
    # # img shape: (288, 512, 3)
    # transform = dict(
    #     type='Resize', img_scale=None, ratio_range=(0.5, 2.0), keep_ratio=True)
    # resize_module = build_from_cfg(transform, PIPELINES)
    # resized_results = resize_module(results.copy())
    # assert int(288 * 0.5) <= resized_results['img_shape'][0] <= 288 * 2.0
    # assert int(512 * 0.5) <= resized_results['img_shape'][1] <= 512 * 2.0


def test_flip():
    # test assertion for invalid prob
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', prob=1.5)
        build_from_cfg(transform, PIPELINES)

    # test assertion for invalid direction
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFlip', prob=1, direction='horizonta')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='RandomFlip', prob=1)
    flip_module = build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    original_img = copy.deepcopy(img)
    seg = np.array(
        Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = flip_module(results)

    flip_module = build_from_cfg(transform, PIPELINES)
    results = flip_module(results)
    assert np.equal(original_img, results['img']).all()
    assert np.equal(original_seg, results['gt_semantic_seg']).all()

def test_pad():
    # test assertion if both size_divisor and size is None
    with pytest.raises(AssertionError):
        transform = dict(type='Pad')
        build_from_cfg(transform, PIPELINES)

    transform = dict(type='Pad', size_divisor=10)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = transform(results)
    # original img already divisible by 10
    assert np.equal(results['img'], original_img).all()
    img_shape = results['img'].shape
    assert img_shape[0] % 10 == 0
    assert img_shape[1] % 10 == 0

    transform = dict(type='Pad', size_divisor=32)
    transform = build_from_cfg(transform, PIPELINES)
    results = dict()
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    original_img = copy.deepcopy(img)
    results['img'] = img
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    results = transform(results)
    # Divise by 32
    img_shape = results['img'].shape
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0

    resize_transform = dict(
        type='Resize', img_scale=(512, 512), keep_ratio=False)
    resize_module = build_from_cfg(resize_transform, PIPELINES)
    results = resize_module(results)
    results = transform(results)
    img_shape = results['img'].shape
    assert img_shape[0] % 512 == 0
    assert img_shape[1] % 512 == 0

    resize_transform = dict(
        type='Resize', img_scale=(512, 512), keep_ratio=True)
    resize_module = build_from_cfg(resize_transform, PIPELINES)
    results = resize_module(results)
    results = transform(results)
    img_shape = results['img'].shape
    # Test resize with keep_ratio
    assert img_shape[0] % 32 == 0
    assert img_shape[1] % 32 == 0


def test_fisheye_crop():
    # with pytest.raises(AssertionError):
    #     transform = dict(type='RandomFisheyeCrop', cat_max_ratio=1., ignore_index=255,
    #                     mvx = 100, const_crop_y = 150, rand_crop_y = 50,
    #                     crop_prob = 0.5, shift_prob = 0.5)
    #     build_from_cfg(transform, PIPELINES)

    results = dict()
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    seg = np.array(Image.open(PATH_TO_SEG))
    results['img'] = img
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    # Set initial values for default meta_keys
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0

    h, w, _ = img.shape
    transform = dict(type='RandomFisheyeCrop', cat_max_ratio=1., ignore_index=255,
                        mvx = 100, const_crop_y = 150, rand_crop_y = 50,
                        crop_prob = 0., shift_prob = 0.)
    crop_module = build_from_cfg(transform, PIPELINES)
    results = crop_module(results)
    assert results['img'].shape[:2] == (h, w)
    # assert results['img'].shape[:2] == (h - 20, w - 20)
    # assert results['img_shape'][:2] == (h - 20, w - 20)
    # assert results['gt_semantic_seg'].shape[:2] == (h - 20, w - 20)
