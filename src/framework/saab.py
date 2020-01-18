# v2020.01.18 

# Saab transformation
# modeiled from https://github.com/davidsonic/Interpretable_CNN

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from numpy import linalg as LA
from skimage.measure import block_reduce
import pickle
import time

class Saab():
    def __init__(self, pca_name, num_kernels, useDC=True, batch=None, needBias=True):
        self.pca_name = pca_name
        self.num_kernels = num_kernels
        self.useDC = useDC
        self.batch = batch
        self.needBias = needBias

    def remove_mean(self, feature, axis):
        feature_mean = np.mean(feature, axis=axis, keepdims=True)
        feature = feature - feature_mean
        return feature, feature_mean

    def Transform(self, feature, kernels):
        if self.batch == None:
            transformed = np.matmul(feature, np.transpose(kernels))
        else:
            transformed = []
            for i in range(0,feature.shape[0], self.batch):
                if i+self.batch <= feature.shape[0]:
                    transformed.append(np.matmul(feature[i:i+self.batch], np.transpose(kernels)))
                else:
                    transformed.append(np.matmul(feature[i:], np.transpose(kernels)))
            feature = []
            transformed = np.array(transformed)
            transformed = transformed.reshape(-1,transformed.shape[2],transformed.shape[3],transformed.shape[4])
        return transformed

    def Saab_transform(self, pixelhop_feature, train=True, pca_params=None): 
        if train == True:
            pca_params = {}
            X, no = self.remove_mean(pixelhop_feature.copy(), axis=0)
            X, dc = self.remove_mean(pixelhop_feature.copy(), axis=1)
            pca = PCA(n_components=self.num_kernels, svd_solver='full').fit(X)
            kernels = pca.components_
            energy = pca.explained_variance_ / np.sum(pca.explained_variance_)
            if self.useDC == True:  
                largest_ev = np.var(dc * np.sqrt(X.shape[-1]))     
                dc_kernel = 1 / np.sqrt(pixelhop_feature.shape[-1]) * np.ones((1, pixelhop_feature.shape[-1])) / np.sqrt(largest_ev)
                kernels = np.concatenate((dc_kernel, kernels[:-1]), axis=0)
                energy = np.concatenate((np.array([largest_ev]),pca.explained_variance_[:-1]), axis=0)
                energy = energy/np.sum(energy)
            bias = LA.norm(pixelhop_feature, axis=1)
            bias = np.max(bias)
            pca_params['Kernels'] = kernels
            pca_params['Energy'] = energy
            pca_params['Bias'] = bias
        else:
            kernels = pca_params['Kernels']
            energy = pca_params['Energy']
            bias = pca_params['Bias']
        if self.needBias == True:
            pixelhop_feature += bias
        transformed = self.Transform(pixelhop_feature, kernels) 
        if self.needBias == True:
            e = np.zeros((1, kernels.shape[0]))
            e[0, 0] = 1
            transformed -= bias*e
        return transformed, pca_params

    def fit(self, pixelhop_feature, train=True):
        #print("------------------- Start: Saab transformation")
        #t0 = time.time()
        pca_params = {}
        if train == False:
            fw = open(self.pca_name, 'rb')
            pca_params = pickle.load(fw)
            fw.close()
        transformed, params = self.Saab_transform(pixelhop_feature=pixelhop_feature, train=train, pca_params=pca_params)
        if train == True:
            fw = open(self.pca_name, 'wb')
            pickle.dump(params, fw)
            fw.close()
            #print("       <Info>        Save pca params as name: %s"%str(self.pca_name))
        #print("------------------- End: Saab transformation -> using %10f seconds"%(time.time()-t0))    
        return transformed

