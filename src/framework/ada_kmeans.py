# v2019.11.12.v1
import os
import sys
import numpy as np
import time
import pickle
import scipy
import sklearn
import math
from sklearn import preprocessing 
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import log_loss as LL
from collections import Counter
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# method of leaf node regression
def Regression_Method(X, Y, num_class):
    return LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial',class_weight='balanced').fit(X, Y.reshape(-1,))

# check entropy of meet the limits
def Continue_split(H, limit):
    if H<limit:
        return False
    else:
        return True

# only for binary cases
# used when computing entropy in <Multi_Trial>
def Compute_Weight(Y):
    weight = np.zeros((2))
    for i in range(0,weight.shape[0]):
        weight[i] = 1-(float)(Y[Y==i].shape[0])/(float)(Y.shape[0])
    return weight

###################################################################################

# latest cross entropy method
def Comupte_Cross_Entropy(X, Y, num_class, num_bin=32):
    samp_num = Y.size
    if np.unique(Y).shape[0] == 1: #alread pure
        return 0
    kmeans = KMeans(n_clusters=num_bin, random_state=0).fit(X)
    prob = np.zeros((num_bin, num_class))
    for i in range(num_bin):
        idx = (kmeans.labels_ == i)
        tmp = Y[idx]
        for j in range(num_class):
            prob[i, j] = (float)(tmp[tmp == j].shape[0]) / ((float)(Y[Y==j].shape[0]) + 1e-5)
    prob = (prob - np.min(prob, axis=1).reshape(-1,1))/(np.max(prob, axis=1).reshape(-1,1) - np.min(prob, axis=1).reshape(-1,1) + 1e-5)
    true_indicator = np.zeros((samp_num, num_class))
    true_indicator[np.arange(samp_num), Y] = 1
    probab = prob[kmeans.labels_]
    return LL(true_indicator,probab)/math.log(num_class)

# init kmeans centroid with center of samples from each label, then do kmeans
def Init_By_Class(X, Y, num_class):
    init_centroid = []
    for i in range(np.unique(Y).shape[0]):
        Y = Y.reshape(-1)
        init_centroid.append(np.mean(X[Y==i],axis=0).reshape(-1))
    kmeans = KMeans(n_clusters=np.unique(Y).shape[0], n_jobs=10, init=np.array(init_centroid)).fit(X)
    data = []
    H = []
    Hidx = [0]
    for i in range(np.unique(Y).shape[0]):
        data.append( {'Data': X[kmeans.labels_ == i],
                'Label': Y[kmeans.labels_ == i],
                'Centroid': kmeans.cluster_centers_[i],
                'H': Comupte_Cross_Entropy(X[kmeans.labels_ == i], Y[kmeans.labels_ == i], num_class),
                'ID': str(i)})
        H.append(data[i]['H'])
        Hidx.append(Hidx[-1]+1)
    return data, H, Hidx

# init whole data as one cluster
def Init_As_Whole(X, Y, num_class):
    data = [{'Data':X, 
            'Label':Y, 
            'Centroid':np.mean(X,axis=0), 
            'H':Comupte_Cross_Entropy(X, Y, num_class),
            'ID':'0'}]
    H = [data[0]['H']]
    Hidx = [0,1]
    return data, H, Hidx

# try multiply times when spliting the leaf node
def Multi_Trial(X, sep_num=2, batch_size=None, trial=6, num_class=2):
    init = ['k-means++','random','k-means++','random','k-means++','random']
    H = X['H']
    center = []
    flag = 0
    for i in range(trial):
        if batch_size == None:
            kmeans = KMeans(n_clusters=sep_num, n_jobs=10, init=init[i%6]).fit(X['Data'])
        else:
            kmeans = MiniBatchKMeans(n_clusters=sep_num, batch_size=batch_size).fit(X['Data'])
        # early stop
        k_labels = kmeans.labels_
        counting = np.array(Counter(k_labels.tolist()).most_common(np.unique(k_labels).size))[:,1]
        if np.min(counting)>int(0.05*X['Data'].shape[0]):
            weight = Compute_Weight(kmeans.labels_)
            tH = 0.0
            for k in range(sep_num):
                tH += weight[k]*Comupte_Cross_Entropy(X['Data'][kmeans.labels_ == k], X['Label'][kmeans.labels_ == k], num_class)
            if tH < H:
                H = tH
                center = kmeans.cluster_centers_.copy()
                label = kmeans.labels_.copy()
                flag = 1
                print("           <Info>        Multi_Trial %s: Found a separation better than original! CE: %s"%(str(i),str(H)))
    if flag == 0:
        return []
    subX = []
    for i in range(sep_num):
        idx = (label == i)
        subX.append({'Data':X['Data'][idx], 'Label':X['Label'][idx], 'Centroid':center[i], 'H':Comupte_Cross_Entropy(X['Data'][idx],X['Label'][idx], num_class),'ID':X['ID']+str(i)})
    return subX 

def Leaf_Node_Regression(data, Hidx, num_class):
    for i in range(len(Hidx)-1):
        data[Hidx[i]]['Regressor'] = Regression_Method(X, Y, num_class)
    return data

def Ada_KMeans(X, Y, trial=6, batch_size=10000, minS=0.1, maxN=50, limit=0.5, maxiter=50):
    # trial: # of runs in each separation
    # minS: minimum percent of samples in each cluster
    # maxN: max number of leaf nodes (centroids)
    # limit: stop splitting when the max CE<limit
    # max iteration
    print("=========== Start: Ada_KMeans")
    t0 = time.time()
    print("       <Info>        Input shape: %s"%str(X.shape))
    print("       <Info>        Trial: %s"%str(trial))
    print("       <Info>        Batch size: %s"%str(batch_size))
    print("       <Info>        Minimum percent of samples in each cluster: %s"%str(minS))
    print("       <Info>        Max number of leaf nodes: %s"%str(batch_size))
    print("       <Info>        Stop splitting when the max CE<limit: %s"%str(limit))
    print("       <Info>        Max iteration: %s"%str(maxiter))
    # H: <list> entropy of nodes can be split
    # Hidx: <list> location of corresponding H in data
    num_class = np.unique(Y).shape[0]
    num_sample = X.shape[0]
    data, H, Hidx = Init_By_Class(X, Y, num_class)
    X, Y = [], []
    N ,myiter = 1, 1
    print("\n       <Info>        Start iteration")
    while N < maxN and myiter < maxiter:
        print("       <Info>        Iter %s"%(str(myiter)))
        idx = np.argmax(np.array(H))
        if Continue_split(H[idx], limit) == False: # continue to split?
            print("       <Info>        Finish splitting!")
            break
        if data[Hidx[idx]]['Data'].shape[0] < int(minS*num_sample): # if this cluster has too few sample, change the next largest
            print("       <Warning>        Iter %s: Too small! continue for the next largest"%str(myiter))
            H[idx] = -H[idx]
            continue
        subX = Multi_Trial(data[Hidx[idx]], batch_size=batch_size, trial=trial, num_class=num_class)
        if len(subX)!=0:
            # save memory, do not save X, Y multi times
            data[Hidx[idx]]['Data'] = []
            data[Hidx[idx]]['Label'] = []
            data += subX
            H.pop(idx)
            Hidx.pop(idx)
            N -= 1
            for d in subX:
                H.append(d['H'])
                Hidx.append(Hidx[-1]+1)
                N += 1
            myiter += 1
        else:
            print("       <Warning>        Iter %s: Don't split! continue for the next largest"%str(myiter))
            H[idx] = -H[idx]
    data = Leaf_Node_Regression(data, Hidx, num_class)
    print("=========== End: Ada_KMeans -> using %10f seconds"%(time.time()-t0))
    return data

'''
def Merge(Y, data, Hidx):
    centroid = []
    Y = []
    cluster_label = []
    n0 = 9
    n1 = 1
    for i in range(0, len(Hidx)-1):
        Y.append(data[Hidx[i]]['Label'])
        centroid.append(data[Hidx[i]]['Centroid'].reshape(-1))
        t = data[Hidx[i]]['Label']
        if t[t == 0].shape[0]/n0 > t[t == 1].shape[0]/n1:
            cluster_label.append(0)
        elif t[t == 0].shape[0]/n0 < t[t == 1].shape[0]/n1:
            cluster_label.append(1)
        if i == 0:
            X = data[Hidx[i]]['Data']
            label = i*np.ones((data[Hidx[i]]['Label'].shape[0],1))
        else:
            X = np.concatenate((X,data[Hidx[i]]['Data']),axis=0)
            label = np.concatenate((label, i*np.ones((data[Hidx[i]]['Label'].shape[0],1))), axis=0)    
    Y = np.array(Y)
    Y = Y.reshape(-1,1)
    centroid = np.array(centroid)
    cluster_label = np.array(cluster_label)
    cluster_label = cluster_label.reshape(-1,1)
    return X, Y, label, centroid, cluster_label

'''

if __name__ == "__main__":
    import cv2
    X = cv2.imread('../../data/test.jpg')
    X = cv2.resize(X, (40,40))
    X = X.reshape(-1,3)
    Y = np.random.randint(2,size=X.shape[0])
    '''
    print(" \n> This is a test enample: ")
    X = np.array([[-1, -1, 1], [-1, -2, 1], [-2, -1, 1], [-2, -2, 1], [1, 1, 5], [2, 3, 4]])
    Y = np.array([0, 0, 0, 1, 1, 1])
    '''
    Y = Y.reshape(-1,1)
    #print(" \n--> Input X... \n", X)
    #print(" \n--> Input Y... \n", Y)
    data = Ada_KMeans(X, Y, limit=1, maxN=10)
    print(" \n--> Result centroids... \n", data)
    