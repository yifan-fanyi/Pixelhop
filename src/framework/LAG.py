#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:49:47 2019

@author: yueru
"""
# modified by Alex 2019.10.11

import os
import numpy as np
import time
import pickle
import scipy
from sklearn import preprocessing 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import MiniBatchKMeans
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_target_(X, Y, num_clusters, class_list, batch_size= 1000): 
    Y = Y.reshape(-1)
    num_clusters_sub = int(num_clusters/len(class_list))
    labels = np.zeros((X.shape[0]))
    clus_labels = np.zeros((num_clusters,))
    centroid = np.zeros((num_clusters, X.shape[1]))
    for i in range(len(class_list)):
        ID = class_list[i]
        feature_train = X[Y==ID]
        kmeans = MiniBatchKMeans(n_clusters=num_clusters_sub, batch_size=batch_size).fit(feature_train)
        labels[Y==ID] = kmeans.labels_ + i*num_clusters_sub
        clus_labels[i*num_clusters_sub:(i+1)*num_clusters_sub] = ID
        centroid[i*num_clusters_sub:(i+1)*num_clusters_sub] = kmeans.cluster_centers_
        print ("       <Info>        FINISH KMEANS: %s"%str(i))
    return labels, clus_labels.astype(int), centroid

def llsr_train(X, Y, encode=True, num_clusters=10, class_list=None, alpha=10):
    SAVE = {} 
    labels_train, clus_labels, centroid = compute_target_(X, Y, num_clusters, class_list)    
    scaler = preprocessing.StandardScaler().fit(X)  

    if encode:
        labels_train_onehot = np.zeros((labels_train.shape[0], clus_labels.shape[0]))
        for i in range(labels_train.shape[0]):
            gt = Y[i]
            idx = clus_labels == gt
            dis = euclidean_distances(X[i].reshape(1,-1), centroid[idx]).reshape(-1)
            dis = dis/(dis.min()+1e-5)
            p_dis = np.exp(-dis*alpha)
            p_dis = p_dis/p_dis.sum()
            labels_train_onehot[i,idx] = p_dis            
    else:     
        labels_train_onehot = labels_train 
    feature = scaler.transform(X)
    A = np.ones((feature.shape[0], 1))
    feature = np.concatenate((A, feature), axis=1)
    weight = scipy.linalg.lstsq(feature, labels_train_onehot)[0]       
    weight_save = weight[1:weight.shape[0]]
    bias_save = weight[0].reshape(1, -1)

    SAVE['clus_labels'] = clus_labels
    SAVE['LLSR weight'] = weight_save
    SAVE['LLSR bias'] = bias_save
    SAVE['scaler'] = scaler
    return SAVE

def llsr_acc(feature, Y, SAVE):
    pred_labels = np.zeros((feature.shape[0], len(np.unique(Y))))
    for km_i in range(len(np.unique(Y))):
        pred_labels[:,km_i] = feature[:, SAVE['clus_labels']==km_i].sum(1)
    pred_labels = np.argmax(pred_labels, axis=1)
    idx = pred_labels == Y.reshape(-1)
    print("       <Info>        KMeans train accuracy: %s"%str(1.*np.count_nonzero(idx)/Y.shape[0]))

def llsr_test(X, SAVE=None, weight_path=None):
    if SAVE == None:
        print("       <Info>        load weight: %s"%('../weight/'+weight_path))
        fr = open('../weight/'+weight_path, 'rb')
        SAVE = pickle.load(fr)
        fr.close()
    X = SAVE['scaler'].transform(X)
    X = np.matmul(X, SAVE['LLSR weight'])
    X = X + SAVE['LLSR bias']
    return X

def LAG_Unit(X, Y=None, class_list=None, weight_path="LAG_weight.pkl", num_clusters=50, alpha=5, train=True):
#                  feature: training features or testing features
#                  class_list: list of class labels
#                  SAVE: store parameters
#                  num_clusters: output feature dimension
#                  alpha: a parameter when computing probability vector
#                  Train: True: training stage; False: testing stage
    print("=========== Start: LAG_Unit")
    print("       <Info>        Input feature shape: %s"%str(X.shape))
    print("       <Info>        Class list: %s"%str(class_list))
    print("       <Info>        number of cluster: %s"%str(num_clusters))
    print("       <Info>        alpha: %s"%str(alpha))
    t0 = time.time()
    if train:
        print("------------------- Start: LAG Train")
        t1 = time.time()
        SAVE = llsr_train(X, Y, encode=True, num_clusters=num_clusters, class_list=class_list, alpha=alpha)  
        feature = llsr_test(X, SAVE=SAVE)
        llsr_acc(feature, Y, SAVE)
        fr = open('../weight/'+weight_path, 'wb')
        pickle.dump(SAVE, fr)
        fr.close()
        print("       <Info>        save weight: %s"%('../weight/'+weight_path))
        print("------------------- End: LAG Train -> using %10f seconds"%(time.time()-t1))  
    else:
        feature =llsr_test(X, weight_path=weight_path)
    print("=========== End: LAG_Unit -> using %10f seconds"%(time.time()-t0))
    return feature     