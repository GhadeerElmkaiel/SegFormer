import cv2
import numpy as np
import os

x = np.array([1,2,4])
y = np.array([0.5,1,3])

inv_mean = np.array(-(x/y))
std = 1/y
print(std)