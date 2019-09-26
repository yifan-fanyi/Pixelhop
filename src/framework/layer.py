# Alex
# yifanwang0916@outlook.com
# 2019.09.25

import numpy as np 
import math
import cv2
from skimage.measure import block_reduce

def myResize(x, H, W):
    new_x = np.zeros((x.shape[0], H, W, x.shape[3]))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[3]):
            new_x[i,:,:,j] = cv2.resize(x[i,:,:,j], (W,H), interpolation=cv2.INTER_CUBIC)
    return new_x

def MaxPooling(x):
    return block_reduce(x, (1, 2, 2, 1), np.max)

def AvgPooling(x):
    return block_reduce(x, (1, 2, 2, 1), np.mean)

def Project_concat(feature):
    dim = 0
    for i in range(len(feature)):
        dim += feature[i].shape[3]
        feature[i] = np.moveaxis(feature[i],0,2)
    result = np.zeros((feature[0].shape[0],feature[0].shape[1],feature[0].shape[2],dim))
    for i in range(0,feature[0].shape[0]):
        for j in range(0,feature[0].shape[1]):
            scale = 1.
            for fea in feature:
                if scale == 1:
                    tmp = fea[i,j]
                else:
                    #print(i,j,i//scale,j//scale)
                    tmp = np.concatenate((tmp, fea[int(i//scale),int(j//scale)]), axis=1)
                scale *= 2
            result[i,j] = tmp
    result = np.moveaxis(result, 2, 0)
    return result