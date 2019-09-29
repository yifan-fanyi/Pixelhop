#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:49:47 2019

@author: yueru
"""

import scipy
from sklearn import preprocessing 
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from keras import backend as K
from sklearn.cluster import MiniBatchKMeans

K.tensorflow_backend._get_available_gpus()

def Norm(feature): 
    #   shape:(# sample, # feature)
    feature = feature - np.min(feature,1).reshape(-1,1)
    feature = feature/np.sum(feature,1).reshape(-1,1)
    return feature
def Relu(centroid_old): 
    #   shape:(# sample, # feature)
    centroid_old[centroid_old<0]=0
    return centroid_old
def llsr_train(feature_train,labels_train,encode=True,centroid=None,clus_labels=None,train_labels=None,scaler=None, alpha=10):
    if encode:
        alpha=alpha
        print('Alpha:',alpha)
#        labels_train_onehot = encode_onehot(labels_train) 
        n_sample=labels_train.shape[0]
        labels_train_onehot=np.zeros((n_sample,clus_labels.shape[0]))
        for i in range(n_sample):
            gt=train_labels[i]
            idx=clus_labels==gt
            dis=euclidean_distances(feature_train[i].reshape(1,-1),centroid[idx]).reshape(-1)
            dis=dis/(dis.min()+1e-5)
            p_dis=np.exp(-dis*alpha)
            p_dis=p_dis/p_dis.sum()
            labels_train_onehot[i,idx]=p_dis            
    else:     
        labels_train_onehot = labels_train 
    feature_train = scaler.transform(feature_train)
    A = np.ones((feature_train.shape[0], 1))
    feature_train = np.concatenate((A, feature_train), axis=1)
    #    print(np.sort(labels_train_onehot[:10],1)[:,::-1])
    weight=scipy.linalg.lstsq(feature_train,labels_train_onehot)[0]       
    weight_save = weight[1:weight.shape[0]]
    bias_save = weight[0].reshape(1, -1)
    return weight_save, bias_save
def llsr_test(feature_test,weight_save,bias_save):
    
    feature_test=np.matmul(feature_test,weight_save)
    feature_test=feature_test+bias_save
    return feature_test

def compute_target_(feature,train_labels,num_clusters,class_list): 
    use_classes=len(class_list) 

    train_labels = train_labels.reshape(-1)
    num_clusters_sub = int(num_clusters/use_classes)
    batch_size= 1000 
    labels = np.zeros((feature.shape[0]))
    clus_labels = np.zeros((num_clusters,))
    centroid = np.zeros((num_clusters, feature.shape[1]))
    for i in range(use_classes):
        ID=class_list[i]
        feature_train = feature[train_labels==ID]
        kmeans = MiniBatchKMeans(n_clusters=num_clusters_sub,batch_size=batch_size).fit(feature_train)
#            kmeans = KMeans(n_clusters=num_clusters_sub).fit(feature_train)
        labels[train_labels==ID] = kmeans.labels_ + i*num_clusters_sub
        clus_labels[i*num_clusters_sub:(i+1)*num_clusters_sub] = ID
        centroid[i*num_clusters_sub:(i+1)*num_clusters_sub] = kmeans.cluster_centers_
        print ('FINISH KMEANS', i)
        
    return labels, clus_labels.astype(int),centroid

def encode_onehot(a):
    a = a.reshape(-1)
    print ('before encode shape:', a.shape)
    b = np.zeros((a.shape[0],1+ int(a.max())))# - 1./a.max()
    b[np.arange(a.shape[0]), a] = 1
    print ('after encode shape:', b.shape)
    return b.astype(float)


        
def LAG_Unit(feature,train_labels=None, class_list=None, SAVE=None,num_clusters=50,alpha=5,Train=True):
#                  feature: training features or testing features
#                  class_list: list of class labels
#                  SAVE: store parameters
#                  num_clusters: output feature dimension
#                  alpha: a parameter when computing probability vector
#                  Train: True: training stage; False: testing stage
                  
    if Train:
            
        print('--------Train LAG Unit--------')    
        print ('feature_train shape:', feature.shape)
        use_classes=len(np.unique(train_labels)) 
        k=0                        
        # Compute output features       
        labels_train,clus_labels,centroid = compute_target_(feature,train_labels,num_clusters,
          class_list)    
#                SAVE['train_dis']=cosine_similarity(feature_train,centroid)
#                SAVE['test_dis']=cosine_similarity(feature_test,centroid)
        # Solve LSR
        scaler=preprocessing.StandardScaler().fit(feature)  
        weight_save,bias_save=llsr_train(feature,labels_train.astype(int),encode=True,centroid=centroid,
                                         clus_labels=clus_labels,train_labels=train_labels,
                                         scaler=scaler,alpha=alpha)
        
        SAVE[str(k)+' clus_labels'] = clus_labels
        SAVE[str(k)+' LLSR weight'] = weight_save
        SAVE[str(k)+' LLSR bias'] = bias_save
        SAVE[str(k)+' scaler'] = scaler
        
        feature = llsr_test(scaler.transform(feature),weight_save,bias_save)
        pred_labels = np.zeros((feature.shape[0],use_classes))
        for km_i in range(use_classes):
            pred_labels[:,km_i]=feature[:,clus_labels==km_i].sum(1)
        pred_labels = np.argmax(pred_labels, axis=1)
        idx = pred_labels == train_labels.reshape(-1)
        print(k, ' Kmean training acc is: {}'.format(1.*np.count_nonzero(idx)/train_labels.shape[0]))
        return feature
        
    else:
        print('--------Testing--------')
        k=0
        scaler=SAVE[str(k)+' scaler']
        feature_reduced =llsr_test(scaler.transform(feature),SAVE[str(k)+' LLSR weight'],SAVE[str(k)+' LLSR bias'])
        return feature_reduced      
    

