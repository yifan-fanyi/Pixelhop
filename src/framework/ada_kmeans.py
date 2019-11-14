# v2019.11.12.v2
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
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# method of leaf node regression
def Regression_Method(X, Y, num_class):
    return LogisticRegression(random_state=0, solver='newton-cg', multi_class='ovr',class_weight='balanced', n_jobs=20).fit(X, Y.reshape(-1,))

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

def Compute_GlobalH(X, total, Hidx):
    gH = 0.0
    for i in range(len(Hidx)-1):
        gH += (X[Hidx[i]]['Data'].shape[0]/float(total))*X[Hidx[i]]['H']
    return gH

def Draw_globalH(globalH):
    print("drawing meanCE...")
    plt.figure(0)
    plt.plot(globalH,'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Conditional Cross Entropy')
    plt.xticks(range(len(globalH)))
    plt.savefig('./meanCE_hop'+str(time.time())+'.png')
    plt.close(0)
###################################################################################

# latest cross entropy method
def Comupte_Cross_Entropy(X, Y, num_class, num_bin=32):
    samp_num = Y.size
    if np.unique(Y).shape[0] == 1: #alread pure
        return 0
    if X.shape[0] < num_bin:
        return -1
    kmeans = KMeans(n_clusters=num_bin, random_state=0).fit(X)
    prob = np.zeros((num_bin, num_class))
    for i in range(num_bin):
        idx = (kmeans.labels_ == i)
        tmp = Y[idx]
        for j in range(num_class):
            prob[i, j] = (float)(tmp[tmp == j].shape[0]) / ((float)(Y[Y==j].shape[0]) + 1e-5)
    prob = (prob)/(np.sum(prob, axis=1).reshape(-1,1) + 1e-5)
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
        data.append({'Data': X[kmeans.labels_ == i],
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
        data[Hidx[i]]['Regressor'] = Regression_Method(data[Hidx[i]]['Data'], data[Hidx[i]]['Label'], num_class)
        data[Hidx[i]]['Data'] = [] #no need to store raw data any more
        data[Hidx[i]]['Label'] = []
    return data

def Ada_KMeans_train(X, Y, sep_num=2, trial=6, batch_size=10000, minS=300, maxN=50, limit=0.5, maxiter=50):
    # trial: # of runs in each separation
    # minS: minimum number of samples in each cluster
    # maxN: max number of leaf nodes (centroids)
    # limit: stop splitting when the max CE<limit
    # max iteration
    print("------------------- Start: Ada_KMeans_train")
    t0 = time.time()
    print("       <Info>        Trial: %s"%str(trial))
    print("       <Info>        Batch size: %s"%str(batch_size))
    print("       <Info>        Minimum number of samples in each cluster: %s"%str(minS))
    print("       <Info>        Max number of leaf nodes: %s"%str(batch_size))
    print("       <Info>        Stop splitting when the max CE<limit: %s"%str(limit))
    print("       <Info>        Max iteration: %s"%str(maxiter))
    # H: <list> entropy of nodes can be split
    # Hidx: <list> location of corresponding H in data
    num_class = np.unique(Y).shape[0]
    data, H, Hidx = Init_By_Class(X, Y, num_class)
    rootSampNum = Y.shape[0]
    global_H = []

    X, Y = [], []
    N ,myiter = 1, 1
    print("\n       <Info>        Start iteration")
    while N < maxN and myiter < maxiter+1:
        print("       <Info>        Iter %s"%(str(myiter)))
        idx = np.argmax(np.array(H))
        if Continue_split(H[idx], limit) == False: # continue to split?
            print("       <Info>        Finish splitting!")
            break
        if data[Hidx[idx]]['Data'].shape[0] < minS: # if this cluster has too few sample, change the next largest
            print("       <Warning>        Iter %s: Too small! continue for the next largest"%str(myiter))
            H[idx] = -H[idx]
            continue
        subX = Multi_Trial(data[Hidx[idx]], sep_num=sep_num, batch_size=batch_size, trial=trial, num_class=num_class)
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
            global_H.append(Compute_GlobalH(data,rootSampNum,Hidx))
        else:
            print("       <Warning>        Iter %s: Don't split! continue for the next largest"%str(myiter))
            H[idx] = -H[idx]
    data = Leaf_Node_Regression(data, Hidx, num_class)
    print("------------------- End: Ada_KMeans_train -> using %10f seconds"%(time.time()-t0))
    return data, global_H        

# list to dictionary
def List2Dict(data):
    res = {}
    for i in range(len(data)):
        res[data[i]['ID']] = data[i]
    return res

def Ada_KMeans_Iter_test(X, key_parent, data, sep_num=2):
    centroid = []
    key_child = []
    for i in range(sep_num):
        centroid.append(data[key_parent+str(i)]['Centroid'].reshape(-1))
        key_child.append(key_parent+str(i))
    centroid = np.array(centroid)
    dist = euclidean_distances(X.reshape(1,-1), centroid).squeeze()
    key = key_child[np.argmin(dist)]
    if 'Regressor' in data[key]:
        return key   
    return Ada_KMeans_Iter_test(X, key, data, sep_num)

def Ada_KMeans_test(X, data, sep_num):
    pred = []
    for i in range(X.shape[0]):
        pred.append(data[Ada_KMeans_Iter_test(X[i], '', data, sep_num)]['Regressor'].predict_proba(X[i].reshape(1,-1)))
    return np.array(pred)

def Ada_KMeans(X, Y=None, path='tmp.pkl', train=True, sep_num=2, trial=6, batch_size=10000, minS=300, maxN=50, limit=0.5, maxiter=50):
    print("=========== Start: Ada_KMeans")
    print("       <Info>        Input shape: %s"%str(X.shape))
    print("       <Info>        train: %s"%str(train))
    t0 = time.time()
    if train == True:
        data, globalH = Ada_KMeans_train(X, Y, sep_num=sep_num, trial=trial, batch_size=batch_size, minS=minS, maxN=maxN, limit=limit, maxiter=maxiter)
        data = List2Dict(data)
        f = open('../weight/'+path, 'wb')
        pickle.dump(data, f)
        f.close()
        Draw_globalH(globalH)
    else:
        f = open('../weight/'+path, 'rb')
        data = pickle.load(f)
        f.close()
    X = Ada_KMeans_test(X, data, sep_num)
    print("=========== End: Ada_KMeans_train -> using %10f seconds"%(time.time()-t0))
    return X

if __name__ == "__main__":
    import cv2
    X = cv2.imread('../../data/test.jpg')
    X = cv2.resize(X, (40,40))
    X = X.reshape(-1,3)
    Y = np.random.randint(2, size=X.shape[0])
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
    