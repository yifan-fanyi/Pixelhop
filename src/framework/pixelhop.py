# Alex
# yifanwang0916@outlook.com
# last update 2019.09.25

# PixelHop unit

# feature: <4-D array>, (N, H, W, D)
# dilate: <int> dilate for pixelhop (default: 1)
# num_AC_kernels: <int> AC kernels used for Saab (default: 6)
# pad: <'reflect' or 'none' or 'zeros'> padding method (default: 'reflect)
# weight_name: <string> weight file (in '../weight/'+weight_name) to be saved or loaded. 
# getK: <bool> 0: using saab to get weight; 1: loaded pre-achieved weight
# useDC: <bool> add a DC kernel. 0: not use (out kernel is num_AC_kernels); 1: use (out kernel is num_AC_kernels+1)

# return <4-D array>, (N, H_new, W_new, D_new)

import numpy as np 
import pickle
import time

from framework.saab import Saab

def PixelHop_8_Neighbour(feature, dilate, pad):
    print("------------------- Start: PixelHop_8_Neighbour")
    print("       <Info>        Input feature shape: %s"%str(feature.shape))
    print("       <Info>        dilate: %s"%str(dilate))
    print("       <Info>        padding: %s"%str(pad))
    t0 = time.time()
    S = feature.shape
    if pad == 'reflect':
        feature = np.pad(feature, ((0,0),(dilate, dilate),(dilate, dilate),(0,0)), 'reflect')
    elif pad == 'zeros':
        feature = np.pad(feature, ((0,0),(dilate, dilate),(dilate, dilate),(0,0)), 'constant', constant_values=0)
    if pad == "none":
        res = np.zeros((S[1]-2*dilate, S[2]-2*dilate, S[0], 9*S[3]))
    else:
        res = np.zeros((S[1], S[2], S[0], 9*S[3]))
    idx = np.array([-1, 0, 1])
    feature = np.moveaxis(feature, 0, 2)
    for i in range(dilate, feature.shape[0]-dilate):
        for j in range(dilate, feature.shape[1]-dilate):
            tmp = []
            for ii in idx:
                for jj in idx:
                    iii = i+ii*dilate
                    jjj = j+jj*dilate
                    tmp.append(feature[iii, jjj])
            tmp = np.array(tmp)
            tmp = np.moveaxis(tmp,0,1)
            res[i-dilate, j-dilate] = tmp.reshape(S[0],-1)
    res = np.moveaxis(res, 2, 0)
    print("       <Info>        Output feature shape: %s"%str(res.shape))
    print("------------------- End: PixelHop_8_Neighbour -> using %10f seconds"%(time.time()-t0))
    return res 

def Pixelhop_fit(weight_path, feature, useDC):
    print("------------------- Start: Pixelhop_fit")
    print("       <Info>        Using weight: %s"%str(weight_path))
    t0 = time.time()
    fr = open(weight_path, 'rb')
    pca_params = pickle.load(fr)
    fr.close()
    weight = pca_params['Layer_0/kernel'].astype(np.float32)
    bias = pca_params['Layer_%d/bias' % 0]
    # Add bias
    feature_w_bias = feature + 1 / np.sqrt(feature.shape[3]) * bias
    # Transform to get data for the next stage
    transformed_feature = np.matmul(feature_w_bias, np.transpose(weight))
    if useDC == True:
        e = np.zeros((1, weight.shape[0]))
        e[0, 0] = 1
        transformed_feature -= bias * e
    print("       <Info>        Transformed feature shape: %s"%str(transformed_feature.shape))
    print("------------------- End: Pixelhop_fit -> using %10f seconds"%(time.time()-t0))
    return transformed_feature

def PixelHop_Unit(feature, dilate=1, num_AC_kernels=6, pad='reflect', weight_name='tmp.pkl', getK=False, useDC=False):
    print("=========== Start: PixelHop_Unit")
    t0 = time.time()
    feature = PixelHop_8_Neighbour(feature, dilate, pad)
    if getK == True:
        saab = Saab('../weight/'+weight_name, kernel_sizes=np.array([3]), num_kernels=np.array([num_AC_kernels]), useDC=useDC)
        saab.fit(feature)
    transformed_feature = Pixelhop_fit('../weight/'+weight_name, feature, useDC) 
    print("       <Info>        Output feature shape: %s"%str(transformed_feature.shape))
    print("=========== End: PixelHop_Unit -> using %10f seconds"%(time.time()-t0))
    return transformed_feature



