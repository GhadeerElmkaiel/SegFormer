import cv2
import numpy as np
import os

path = "data/SberMerged_RGBD/train/depth_png/"
path_npy = "data/SberMerged_RGBD/train/depth_npy/"
path_norm = "data/SberMerged_RGBD/train/normalized_depth/"
names = os.listdir(path)
for name in names:
    img_path = path+name
    img = cv2.imread(img_path, -1)

    # print(img)
    vals = np.unique(img)
    print("max val: ",np.max(vals))
    print("min val: ",np.min(vals))
    # print(len(vals))
    break

names = os.listdir(path_npy)
for name in names:
    img_path = path_npy+name
    img = np.load(img_path)
    img = np.nan_to_num(img, 0)

    # print(img)
    vals = np.unique(img)
    print("max val: ",np.max(vals[:-1]))
    print("min val: ",np.min(vals))
    # print(len(vals))
    break

names = os.listdir(path_norm)
for name in names:
    img_path = path_norm+name
    img = np.load(img_path)
    img = np.nan_to_num(img, 0)

    # print(img)
    vals = np.unique(img)
    print("max val: ",np.max(vals))
    print("min val: ",np.min(vals))
    # print(len(vals))
    break