# v2019.11.04 add batch support
# Alex
# yifanwang0916@outlook.com
# last update 2019.10.25

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
    t0 = time.time()
    S = feature.shape
    idx = [-1, 0, 1]
    if pad == 'reflect':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(feature, ((0,0),(dilate[-1], dilate[-1]),(dilate[-1], dilate[-1]),(0,0)), 'constant', constant_values=0)
    if pad == "none":
        dilate = np.array(dilate).astype('int64')
        res = np.zeros((S[1]-2*dilate[-1], S[2]-2*dilate[-1], S[0], (8*dilate.shape[0]+1)*S[3]))
    else:
        dilate = np.array(dilate).astype('int64')
        res = np.zeros((S[1], S[2], S[0], (8*dilate.shape[0]+1)*S[3]))
    feature = np.moveaxis(feature, 0, 2)
    for i in range(dilate[-1], feature.shape[0]-dilate[-1]):
        for j in range(dilate[-1], feature.shape[1]-dilate[-1]):
            tmp = []
            for d in dilate:
                for ii in idx:
                    for jj in idx:
                        if ii == 0 and jj == 0:
                            continue
                        iii = i+ii*d
                        jjj = j+jj*d
                        tmp.append(feature[iii, jjj])
            tmp.append(feature[i,j])
            tmp = np.array(tmp)
            tmp = np.moveaxis(tmp,0,1)
            res[i-dilate[-1], j-dilate[-1]] = tmp.reshape(S[0],-1)
    res = np.moveaxis(res, 2, 0)
    #print("       <Info>        Output feature shape: %s"%str(res.shape))
    #print("------------------- End: PixelHop_Neighbour -> using %10f seconds"%(time.time()-t0))
    return res 

def Batch_PixelHop_Neighbour(feature, dilate, pad, batch):
    res = []
    for i in range(0,feature.shape[0],batch):
        if i+batch <= feature.shape[0]:
            res.append(PixelHop_Neighbour(feature[i:i+batch], dilate, pad))
        else:
            res.append(PixelHop_Neighbour(feature[i:], dilate, pad))
    res = np.array(res)
    print(res.shape)
    res = res.reshape(-1,res.shape[2],res.shape[3], res.shape[4])
    return res

def Pixelhop_fit(weight_path, feature, useDC):
    #print("------------------- Start: Pixelhop_fit")
    #print("       <Info>        Using weight: %s"%str(weight_path))
    t0 = time.time()
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
    res = []
    for i in range(0,feature.shape[0],batch):
        if i+batch <= feature.shape[0]:
            res.append(Pixelhop_fit('../weight/'+weight_name, feature[i:i+batch], useDC))
        else:
            res.append(Pixelhop_fit('../weight/'+weight_name, feature[i:], useDC))
    res = np.array(res)
    res = res.reshape(-1, res.shape[2],res.shape[3],res.shape[4])
    return res

def PixelHop_Unit(feature, dilate=np.array([1]), num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=False, useDC=False, batch=None):
    print("=========== Start: PixelHop_Unit")
    print("       <Info>        Batch size: %s"%str(batch))
    t0 = time.time()
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
    print("       <Info>        Output feature shape: %s"%str(feature.shape))
    print("=========== End: PixelHop_Unit -> using %10f seconds"%(time.time()-t0))
    return feature
