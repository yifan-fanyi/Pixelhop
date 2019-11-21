# v2019.11.21 faster PixelHop_Neighbour
# PixelHop unit

# feature: <4-D array>, (N, H, W, D)
# dilate: <list or np.array> dilate for pixelhop (default: 1)
# num_AC_kernels: <int> AC kernels used for Saab (default: 6)
# pad: <'reflect' or 'none' or 'zeros'> padding method (default: 'reflect)
# weight_name: <string> weight file (in '../weight/'+weight_name) to be saved or loaded. 
# getK: <bool> 0: using saab to get weight; 1: loaded pre-achieved weight
# useDC: <bool> add a DC kernel. 0: not use (out kernel is num_AC_kernels); 1: use (out kernel is num_AC_kernels+1)
# batch: <int/None> minbatch for saving memory 
# return <4-D array>, (N, H_new, W_new, D_new)

import numpy as np 
import pickle
import time

from framework.saab import Saab

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

def Pixelhop_fit(weight_path, feature, useDC):
    #print("------------------- Start: Pixelhop_fit")
    #print("       <Info>        Using weight: %s"%str(weight_path))
    #t0 = time.time()
    fr = open(weight_path, 'rb')
    pca_params = pickle.load(fr)
    fr.close()
    weight = pca_params['Layer_0/kernel'].astype(np.float32)
    bias = pca_params['Layer_%d/bias' % 0]
    # Add bias
    feature = feature + 1 / np.sqrt(feature.shape[3]) * bias
    # Transform to get data for the next stage
    feature = np.matmul(feature, np.transpose(weight))
    if useDC == True:
        e = np.zeros((1, weight.shape[0]))
        e[0, 0] = 1
        feature -= bias * e
    #print("       <Info>        Transformed feature shape: %s"%str(feature.shape))
    #print("------------------- End: Pixelhop_fit -> using %10f seconds"%(time.time()-t0))
    return feature

def Batch_Pixelhop_fit(weight_name, feature, useDC, batch):
    if batch <= feature.shape[0]:
        res = Pixelhop_fit('../weight/'+weight_name, feature[0:batch], useDC)
    else:
        res = Pixelhop_fit('../weight/'+weight_name, feature, useDC)
    for i in range(batch, feature.shape[0], batch):
        if i+batch <= feature.shape[0]:
            res = np.concatenate((res, Pixelhop_fit('../weight/'+weight_name, feature[i:i+batch], useDC)), axis=0)
        else:
            res = np.concatenate((res, Pixelhop_fit('../weight/'+weight_name, feature[i:], useDC)), axis=0)
    return res

def PixelHop_Unit(feature, dilate=np.array([1]), num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=False, useDC=False, batch=None):
    print("=========== Start: PixelHop_Unit")
    print("       <Info>        Batch size: %s"%str(batch))
    t0 = time.time()
    if getK == True:
        if batch == None:
            feature = PixelHop_Neighbour(feature, dilate, pad)
        else:
            feature = Batch_PixelHop_Neighbour(feature, dilate, pad, batch)
        if getK == True:
            saab = Saab('../weight/'+weight_name, num_kernels=num_AC_kernels, useDC=useDC, batch=batch)
            saab.fit(feature)
        if batch == None:
            feature = Pixelhop_fit('../weight/'+weight_name, feature, useDC) 
        else:
            feature = Batch_Pixelhop_fit('../weight/'+weight_name, feature, useDC, batch)
    else:
        if batch == None:
            feature = PixelHop_Neighbour(feature, dilate, pad)
            feature = Pixelhop_fit('../weight/'+weight_name, feature, useDC)
        else:
            if batch <= feature.shape[0]:
                tmp = PixelHop_Neighbour(feature[0:batch], dilate, pad)
                feature_res = Pixelhop_fit('../weight/'+weight_name, tmp, useDC)
            else:
                tmp = PixelHop_Neighbour(feature, dilate, pad)
                feature_res = Pixelhop_fit('../weight/'+weight_name, tmp, useDC)
            for i in range(batch, feature.shape[0], batch):
                if i+batch <= feature.shape[0]:
                    tmp = PixelHop_Neighbour(feature[i:i+batch], dilate, pad)
                    feature_res = np.concatenate((feature_res, Pixelhop_fit('../weight/'+weight_name, tmp, useDC)), axis=0)
                else:
                    tmp = PixelHop_Neighbour(feature[i:], dilate, pad)
                    feature_res = np.concatenate((feature_res, Pixelhop_fit('../weight/'+weight_name, tmp, useDC)), axis=0)
            feature = feature_res
    print("       <Info>        Output feature shape: %s"%str(feature.shape))
    print("=========== End: PixelHop_Unit -> using %10f seconds"%(time.time()-t0))
    return feature
