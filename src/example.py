# Alex
# yifanwang0916@outlook.com
# 2019.09.25

import numpy as np 
import cv2
import time

from framework.layer import *
from framework.utli import *
from framework.pixelhop import *

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances

def myModel(x, getK=1):
    x1 = PixelHop_Unit(x, dilate=1, pad='reflect', num_AC_kernels=9, weight_name='pixelhop1.pkl', getK=getK)

    x2 = PixelHop_Unit(x1, dilate=2, pad='reflect', num_AC_kernels=25, weight_name='pixelhop2.pkl', getK=getK)
    x2 = AvgPooling(x2)

    x3 = PixelHop_Unit(x2, dilate=2, pad='reflect', num_AC_kernels=35, weight_name='pixelhop3.pkl', getK=getK)
    x3 = AvgPooling(x3)

    x4 = PixelHop_Unit(x3, dilate=2, pad='reflect', num_AC_kernels=55, weight_name='pixelhop4.pkl', getK=getK)

    x2 = myResize(x2, x.shape[1], x.shape[2])
    x3 = myResize(x3, x.shape[1], x.shape[2])
    x4 = myResize(x4, x.shape[1], x.shape[2])
    return np.concatenate((x1,x2,x3,x4), axis=3)

x = cv2.imread('../data/test.jpg')
x = x.reshape(1, x.shape[0], x.shape[1], -1)
feature = myModel(x, getK=0)