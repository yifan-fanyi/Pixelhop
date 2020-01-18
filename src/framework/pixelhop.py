# v2020.01.18 
# PixelHop unit

# PixelHop_Unit(feature, dilate=np.array([1]), num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=False, batch=None, needBias=True)
# feature: <4-D array>, (N, H, W, D)
# dilate: <list or np.array> dilate for pixelhop (default: 1)
# num_AC_kernels: <int> AC kernels used for Saab (default: 6)
# pad: <'reflect' or 'none' or 'zeros'> padding method (default: 'reflect)
# weight_name: <string> weight file (in '../weight/'+weight_name) to be saved or loaded. 
# getK: <bool> 0: using saab to get weight; 1: loaded pre-achieved weight
# batch: <int/None> minbatch for saving memory 
# needBias: <bool> 

# return <4-D array>, (N, H_new, W_new, D_new)

import numpy as np 
import pickle
import time

from saab import *

def PixelHop_Neighbour(feature, dilate, pad):
    #print("------------------- Start: PixelHop_Neighbour")
    #print("       <Info>        Input feature shape: %s"%str(feature.shape))
    #print("       <Info>        dilate: %s"%str(dilate))
    #print("       <Info>        padding: %s"%str(pad))
    #t0 = time.time()
    dilate = np.array(dilate)
    idx = [1, 0, -1]
    H, W = feature.shape[1], feature.shape[2]
    res = feature.copy()
    if pad == 'reflect':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'constant', constant_values=0)
    else:
        H, W = H - 2*dilate[-1], W - 2*dilate[-1]
        res = feature[:, dilate[-1]:dilate[-1]+H, dilate[-1]:dilate[-1]+W].copy()
    for d in range(dilate.shape[0]):
        for i in idx:
            for j in idx:
                if i == 0 and j == 0:
                    continue
                else:
                    ii, jj = (i+1)*dilate[d], (j+1)*dilate[d]
                    res = np.concatenate((feature[:, ii:ii+H, jj:jj+W], res), axis=3)
    #print("       <Info>        Output feature shape: %s"%str(res.shape))
    #print("------------------- End: PixelHop_Neighbour -> using %10f seconds"%(time.time()-t0))
    return res 

def Batch_PixelHop_Neighbour(feature, dilate, pad, batch):
    if batch <= feature.shape[0]:
        res = PixelHop_Neighbour(feature[0:batch], dilate, pad)
    else:
        res = PixelHop_Neighbour(feature, dilate, pad)
    for i in range(batch, feature.shape[0], batch):
        if i+batch <= feature.shape[0]:
            res = np.concatenate((res, PixelHop_Neighbour(feature[i:i+batch], dilate, pad)), axis=0)
        else:
            res = np.concatenate((res, PixelHop_Neighbour(feature[i:], dilate, pad)), axis=0)
    return res

def PixelHop_Unit(feature, dilate=np.array([1]), num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=False, batch=None, needBias=True):
    print("=========== Start: PixelHop_Unit")
    print("       <Info>        Input feature shape: %s"%str(feature.shape))
    #print("       <Info>        Batch size: %s"%str(batch))
    t0 = time.time()
    S = feature.shape
    if batch == None:
        feature = PixelHop_Neighbour(feature, dilate, pad)
    else:
        feature = Batch_PixelHop_Neighbour(feature, dilate, pad, batch)
    feature = feature.reshape(-1,feature.shape[-1])
    if getK == True:
        saab = Saab(weight_name, num_kernels=num_AC_kernels, batch=batch, needBias=needBias)
        feature = saab.fit(feature, train=1)
    else:
        saab = Saab(weight_name, num_kernels=num_AC_kernels, batch=batch, needBias=needBias)
        feature = saab.fit(feature, train=0)
    feature = feature.reshape(S[0], S[1], S[2], -1)
    print("       <Info>        Output feature shape: %s"%str(feature.shape))
    print("=========== End: PixelHop_Unit -> using %10f seconds"%(time.time()-t0))
    return feature

if __name__ == "__main__":
    import cv2
    X = cv2.imread('test.jpg')
    X = X.reshape(1, X.shape[0], X.shape[1], X.shape[2]).astype('float64')
    X1 = PixelHop_Unit(X, dilate=np.array([1]), num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=1, batch=None, needBias=True)
    print(X1.shape)
    X2 = PixelHop_Unit(X, dilate=np.array([1]), num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=0, batch=None, needBias=True)
    print(X2.shape)