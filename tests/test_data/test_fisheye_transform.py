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

def test_fisheye_shift_prob_assert_error():
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFisheyeShift', prob=-1)
        shift = build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        transform = dict(type='RandomFisheyeShift', prob=1.5)
        shift = build_from_cfg(transform, PIPELINES)

def test_fisheye_shift_bbox_assert_error():
    with pytest.raises(AssertionError):
        bbox = [[10,10],[0,0]]
        transform = dict(type='RandomFisheyeShift', prob=0.5, bbox=bbox)
        shift = build_from_cfg(transform, PIPELINES)
    with pytest.raises(AssertionError):
        bbox = [[100,100],[100,100]]
        transform = dict(type='RandomFisheyeShift', prob=0.5, bbox=bbox)
        shift = build_from_cfg(transform, PIPELINES)

def test_fisheye_shift_0_prob():
    # Test 0 probability
    transform = dict(type='RandomFisheyeShift', prob=0, max_dx = 100, max_dy = 100)
    shift = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['fisheye_bbox'] = [[0,0], [1440, 1440]]
    for _ in range(30):
        results = shift(results)
        assert np.equal(original_img, results['img']).all()
        assert np.equal(original_seg, results['gt_semantic_seg']).all()


def test_fisheye_shift_bbox_is_equal_to_shape():
    # Test bbox == image.shape
    transform = dict(type='RandomFisheyeShift', prob=1., max_dx = 10, max_dy = 40)
    shift = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['fisheye_bbox'] = [[0,0], [img.shape[1]-1, img.shape[0]-1]]
    for _ in range(30):
        results = shift(results)
        assert original_img.shape == results['img'].shape
        assert original_seg.shape == results['gt_semantic_seg'].shape
        assert np.equal(original_img, results['img']).all()
        assert np.equal(original_seg, results['gt_semantic_seg']).all()

def test_fisheye_shift_only_x():
    # Test shift only in one axis X
    transform = dict(type='RandomFisheyeShift', prob=1., max_dx = 100, max_dy = 0)
    shift = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['fisheye_bbox'] = [[71,124], [1370, 1316]]
    for _ in range(30):
        results = shift(results)
        assert np.equal(original_img[:124,:,:], results['img'][:124,:,:]).all()
        assert np.equal(original_seg[:124,:,:], results['gt_semantic_seg'][:124,:,:]).all()
        assert np.equal(original_img[1317:,:,:], results['img'][1317:,:,:]).all()
        assert np.equal(original_seg[1317:,:,:], results['gt_semantic_seg'][1317:,:,:]).all()

def test_fisheye_shift_only_y():
    # Test shift only in one axis Y
    transform = dict(type='RandomFisheyeShift', prob=1., max_dx = 0, max_dy = 100)
    shift = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['fisheye_bbox'] = [[71,124], [1370, 1316]]
    for _ in range(30):
        results = shift(results)
        assert np.equal(original_img[:,:71,:], results['img'][:,:71,:]).all()
        assert np.equal(original_seg[:,:71,:], results['gt_semantic_seg'][:,:71,:]).all()
        assert np.equal(original_img[:,1371:,:], results['img'][:,1371:,:]).all()
        assert np.equal(original_seg[:,1371:,:], results['gt_semantic_seg'][:,1371:,:]).all()

def test_fisheye_shift_dx_dy_ge_shape():
    # Test dx, dy > image.shape
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    transform = dict(type='RandomFisheyeShift', prob=1.,
                max_dx = img.shape[1]+10, max_dy = img.shape[0]+10)
    shift = build_from_cfg(transform, PIPELINES)
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['fisheye_bbox'] = [[71,124], [1370, 1316]]
    results = shift(results)
    for _ in range(30):
        assert original_img.shape == results['img'].shape
        assert original_seg.shape == results['gt_semantic_seg'].shape

def test_fisheye_shift_bbox_recalculate():
    # Test bbox recalculate
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    transform = dict(type='RandomFisheyeShift', prob=1.,
                dx_range=(20,100),dy_range = (20,100))
    shift = build_from_cfg(transform, PIPELINES)
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results['fisheye_bbox'] = [[71,124], [1370, 1316]]
    original_bbox = copy.deepcopy(results['fisheye_bbox'])
    results = shift(results)
    assert results['fisheye_bbox'] != original_bbox


def test_fisheye_crop_0_prob():
    # Test 0 probability
    bbox = [[0,0], [1440, 1440]]
    transform = dict(type='RandomFisheyeCrop', prob=0, bbox = bbox)
    crop = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    for _ in range(30):
        results = crop(results)
        assert np.equal(original_img, results['img']).all()
        assert np.equal(original_seg, results['gt_semantic_seg']).all()

def test_fisheye_crop_part_equals_bbox():
    # Test part == bbox
    bbox = [[71,124], [1370, 1316]]
    bbox_W = bbox[1][0] - bbox[0][0] + 1
    bbox_H = bbox[1][1] - bbox[0][1] + 1
    transform = dict(type='RandomFisheyeCrop', prob=1., bbox = bbox, part_x_range=(1,1), part_y_range=(1,1))
    crop = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results = crop(results)
    assert results['img'].shape == (bbox_H, bbox_W, 3)
    assert np.equal(original_img[bbox[0][1]:bbox[1][1]+1,bbox[0][0]:bbox[1][0]+1,:], results['img']).all()
    assert np.equal(original_seg[bbox[0][1]:bbox[1][1]+1,bbox[0][0]:bbox[1][0]+1,:], results['gt_semantic_seg']).all()

def test_fisheye_crop_size_after_x_y_range():
    # Test crop size after x, y range
    bbox = [[71,124], [1370, 1316]]
    bbox_W = bbox[1][0] - bbox[0][0] + 1
    bbox_H = bbox[1][1] - bbox[0][1] + 1
    ratio = bbox_W/bbox_H
    transform = dict(type='RandomFisheyeCrop', prob=1., bbox = bbox,
                        part_x_range =(0.6, 1.2), part_y_range = (0.3, 1.7))
    crop = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    original_img = copy.deepcopy(img)
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    for _ in range(30):
        results = dict()
        results['img'] = copy.deepcopy(img)
        results['gt_semantic_seg'] = copy.deepcopy(seg)
        results['seg_fields'] = ['gt_semantic_seg']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results = crop(results)
        assert  np.ceil(bbox_W * 0.6).astype('int') <= results['img'].shape[1] <= np.ceil(bbox_W * 1.2).astype('int')
        assert  np.ceil(bbox_H * 0.3).astype('int') <= results['img'].shape[0] <= np.ceil(bbox_H * 1.7).astype('int')

def test_fisheye_crop_bbox_after_crop_without_size():
    # Test bbox after crop without resize
    bbox = [[71,124], [1370, 1316]]
    bbox_W = bbox[1][0] - bbox[0][0] + 1
    bbox_H = bbox[1][1] - bbox[0][1] + 1
    ratio = bbox_W/bbox_H
    transform = dict(type='RandomFisheyeCrop', prob=1., bbox = bbox, part_x_range=(1,1), part_y_range=(1,1))
    crop = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    results = dict()
    original_img = copy.deepcopy(img)
    results['img'] = img
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    results['gt_semantic_seg'] = seg
    results['seg_fields'] = ['gt_semantic_seg']
    results['img_shape'] = img.shape
    results['ori_shape'] = img.shape
    results['pad_shape'] = img.shape
    results['scale_factor'] = 1.0
    results = crop(results)
    assert results['fisheye_bbox'] == [[0,0],[bbox_W-1, bbox_H-1]]

def test_fisheye_crop_bbox_after_crop_with_dx():
    # Test bbox after crop with dx
    bbox = [[71,124], [1370, 1316]]
    bbox_W = bbox[1][0] - bbox[0][0] + 1
    bbox_H = bbox[1][1] - bbox[0][1] + 1
    ratio = bbox_W/bbox_H
    transform = dict(type='RandomFisheyeCrop', dx_range=(50,100), prob=1., bbox = bbox, part_x_range=(1,1), part_y_range=(1,1))
    crop = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    original_img = copy.deepcopy(img)
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    for _ in range(30):
        results = dict()
        results['img'] = copy.deepcopy(img)
        results['gt_semantic_seg'] = copy.deepcopy(seg)
        results['seg_fields'] = ['gt_semantic_seg']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results = crop(results)
        # The Y coordinate should be constant
        assert results['fisheye_bbox'][0][1] == 0
        assert results['fisheye_bbox'][1][1] == bbox_H-1
        # The X coordinate should be in range
        assert 0 <= results['fisheye_bbox'][0][0] <= 99
        assert bbox_W - 101 <= results['fisheye_bbox'][1][0] <= bbox_W - 1

def test_fisheye_crop_bbox_after_crop_with_dy():
    # Test bbox after crop with dx
    bbox = [[71,124], [1370, 1316]]
    bbox_W = bbox[1][0] - bbox[0][0] + 1
    bbox_H = bbox[1][1] - bbox[0][1] + 1
    ratio = bbox_W/bbox_H
    transform = dict(type='RandomFisheyeCrop', dy_range=(5,70), prob=1., bbox = bbox, part_x_range=(1,1), part_y_range=(1,1))
    crop = build_from_cfg(transform, PIPELINES)
    img = mmcv.imread(PATH_TO_IMAGE, 'color')
    original_img = copy.deepcopy(img)
    seg = np.array(Image.open(PATH_TO_SEG))
    original_seg = copy.deepcopy(seg)
    for _ in range(30):
        results = dict()
        results['img'] = copy.deepcopy(img)
        results['gt_semantic_seg'] = copy.deepcopy(seg)
        results['seg_fields'] = ['gt_semantic_seg']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        results = crop(results)
        # The Y coordinate should be in range
        assert 0 <= results['fisheye_bbox'][0][1] <= 69
        assert bbox_H - 69 <= results['fisheye_bbox'][1][1] <= bbox_H - 1 
        # The X coordinate should be constant
        assert results['fisheye_bbox'][0][0] == 0
        assert results['fisheye_bbox'][1][0] == bbox_W-1

