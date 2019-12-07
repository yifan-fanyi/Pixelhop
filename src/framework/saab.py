# v2019.11.21 

# Saab transformation for PixelHop unit
# modeiled from https://github.com/davidsonic/Interpretable_CNN

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from numpy import linalg as LA
from skimage.measure import block_reduce
import pickle
import time

class Saab():
    def __init__(self, pca_name, num_kernels, energy_percent=None, useDC=False, batch=None, needBias=False):
        self.pca_name = pca_name
        self.num_kernels = num_kernels
        self.useDC = useDC
        self.batch = batch
        self.energy_percent = energy_percent
        self.needBias = False

    # axis=0 batch operation would have something wrong
    # I need to go to bed, leave it for furture
    def remove_mean(self, feature, axis):
        feature_mean = np.mean(feature, axis=axis, keepdims=True)
        if self.batch == None or axis == 0:
            feature = feature - feature_mean
        else:
            self.batch *= 1000
            for i in range(0,feature.shape[0],self.batch):
                if i+self.batch <= feature.shape[0]:
                    if axis == 0:
                        feature[i:i+self.batch] = feature[i:i+self.batch] - feature_mean
                    else:
                        feature[i:i+self.batch] = feature[i:i+self.batch] - feature_mean[i:i+self.batch]
                else:
                    if axis == 0:
                        feature[i:] = feature[i:] - feature_mean
                    else:
                        feature[i:] = feature[i:] - feature_mean[i:]
        return feature, feature_mean

    def find_kernels_pca(self, samples):
        if self.num_kernels:
            if self.batch == None:
                pca = IncrementalPCA(n_components=self.num_kernels, batch_size=self.batch)
            else:
                pca = PCA(n_components=self.num_kernels, svd_solver='full')
            num_components = self.num_kernels
        else:
            if self.batch == None:
                pca = IncrementalPCA(n_components=samples.shape[1], batch_size=self.batch)
            else:
                pca = PCA(n_components=samples.shape[1], svd_solver='full')
        pca.fit(samples)
        if self.energy_percent:
            energy = np.cumsum(pca.explained_variance_ratio_)
            num_components = np.sum(energy < self.energy_percent) + 1
        kernels = pca.components_[:num_components, :]
        mean = pca.mean_
        print("       <Info>        Num of kernels: %d" % num_components)
        print("       <Info>        Energy percent: %f" % np.cumsum(pca.explained_variance_ratio_)[num_components - 1])
        return kernels, mean, pca.explained_variance_ratio_

    def Transform(self, feature, kernels):
        if self.batch == None:
            transformed = np.matmul(feature, np.transpose(kernels))
        else:
            transformed = []
            for i in range(0,feature.shape[0],self.batch):
                if i+self.batch <= feature.shape[0]:
                    transformed.append(np.matmul(feature[i:i+self.batch], np.transpose(kernels)))
                else:
                    transformed.append(np.matmul(feature[i:], np.transpose(kernels)))
            feature = []
            transformed = np.array(transformed)
            transformed = transformed.reshape(-1,transformed.shape[2],transformed.shape[3],transformed.shape[4])
        return transformed

    def Saab_transform(self, pixelhop_feature): 
        S = pixelhop_feature.shape
        print("       <Info>        pixelhop_feature.shape: %s"%str(pixelhop_feature.shape))
        pixelhop_feature = pixelhop_feature.reshape(S[0]*S[1]*S[2],-1)
        pca_params = {}
        pixelhop_feature, feature_expectation = self.remove_mean(pixelhop_feature, axis=0)
        pixelhop_feature, dc = self.remove_mean(pixelhop_feature, axis=1)
        print('       <Info>        training_data.shape: {}'.format(pixelhop_feature.shape))
        kernels, mean, energy_k = self.find_kernels_pca(pixelhop_feature)
        num_channels = pixelhop_feature.shape[1]     
        if self.useDC == True:       
            dc_kernel = 1 / np.sqrt(num_channels) * np.ones((1, num_channels))
            kernels = np.concatenate((dc_kernel, kernels), axis=0)
        if self.needBias == True:
            pixelhop_feature = self.Transform(pixelhop_feature, kernels)
            bias = LA.norm(pixelhop_feature, axis=1)
            bias = np.max(bias)
            pca_params['Layer_%d/bias' % 0] = bias
            print("       <Info>        Transformed shape: %s"%str(pixelhop_feature.shape))
        else:
            pca_params['Layer_%d/bias' % 0] = 0
    
        print("       <Info>        Kernel shape: %s"%str(kernels.shape))
        pca_params['Layer_%d/feature_expectation' % 0] = feature_expectation
        pca_params['Layer_%d/kernel' % 0] = kernels
        pca_params['Layer_%d/pca_mean' % 0] = mean
        pca_params['Layer_%d/pca_energy' % 0] = energy_k
        return pca_params

    def fit(self, pixelhop_feature):
        print("------------------- Start: Saab transformation")
        t0 = time.time()
        pca_params = self.Saab_transform(pixelhop_feature=pixelhop_feature)
        fw = open(self.pca_name, 'wb')
        pickle.dump(pca_params, fw)
        fw.close()
        print("       <Info>        Save pca params as name: %s"%str(self.pca_name))
        print("------------------- End: Saab transformation -> using %10f seconds"%(time.time()-t0))    


