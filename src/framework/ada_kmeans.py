# v2019.12.30.v1
import os
import sys
import numpy as np
import time
import pickle
import scipy
import sklearn
import math
import random
import keras
from sklearn import preprocessing 
from sklearn.cluster import MiniBatchKMeans, KMeans
from collections import Counter
import matplotlib.pyplot as plt

from framework.regression import myRegression
import warnings
warnings.filterwarnings('ignore')

# method of leaf node regression
#from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#RandomForestClassifier(n_estimators=10, max_depth=7, verbose=1, n_jobs=20),
def Regression_Method(X, Y, num_class):
    reg = myRegression(sklearn.linear_model.LogisticRegression(solver='liblinear', multi_class='ovr', n_jobs=20),
                        num_class)
    reg.fit(X, Y)
    reg.score(X, Y)
    return reg

#define which cross entropy to use
def Comupte_Cross_Entropy(X, Y, num_class):
    return ML_Cross_Entropy(X, Y, num_class)
    #return KMeans_Cross_Entropy(X, Y, num_class)
    
def Majority_Vote(Y, mvth):
    new_label = -1
    label = np.unique(Y)
    for i in range(label.shape[0]):
        if Y[Y == label[i]].shape[0] > mvth * (float)(Y.shape[0]):
            new_label = label[i]
            break
    return new_label

# select next leaf node to be splited
# alpha: weight importance
def Select_Next_Split(data, Hidx, alpha):
    t = 0
    idx = 0
    for i in range(0,len(Hidx)-1):
        #tt = data[Hidx[i]]['H']*np.exp(-1*alpha/(float)(data[Hidx[i]]['Data'].shape[0]))
        tt = data[Hidx[i]]['H']*np.log((float)(data[Hidx[i]]['Data'].shape[0]))/np.log(alpha+1)
        if t < tt:
            t = tt
            idx = i
    return idx

################################# Global Cross Entropy #################################
def Compute_GlobalH(X, total, Hidx):
    gH = 0.0
    H = []
    w = []
    for i in range(len(Hidx)-1):
        w.append((X[Hidx[i]]['Data'].shape[0]/float(total)))
        H.append(X[Hidx[i]]['H'])
        gH += (X[Hidx[i]]['Data'].shape[0]/float(total))*X[Hidx[i]]['H']
    print("       <Debug Info>        Emtropy: %s"%str(H))
    print("       <Debug Info>        Weight: %s"%str(w))
    return gH

def Draw_globalH(globalH):
    print("\ndrawing meanCE...")
    plt.figure(0)
    plt.plot(globalH,'bo-')
    plt.xlabel('Iteration')
    plt.ylabel('Conditional Cross Entropy')
    plt.xticks(range(0, len(globalH), len(globalH)//10+1))
    plt.savefig('./meanCE_hop'+str(time.time())+'.png')
    plt.close(0)

################################# Cross Entropy #################################
# used when computing entropy in <Multi_Trial>
def Compute_Weight(Y):
    weight = np.zeros(np.unique(Y).shape[0])
    for i in range(0,weight.shape[0]):
        if (Y[Y==i].shape[0]) == 0:
            weight[i] = 0
        else:
            weight[i] = (float)(Y[Y==i].shape[0])
    weight /= np.sum(weight)
    return weight

# latest cross entropy method
def KMeans_Cross_Entropy(X, Y, num_class, num_bin=32):
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
    true_indicator = keras.utils.to_categorical(Y, num_classes=num_class)
    probab = prob[kmeans.labels_]
    return sklearn.metrics.log_loss(true_indicator,probab)/math.log(num_class)

# new machine learning based cross entropy
def ML_Cross_Entropy(X, Y, num_class):
    X, XX, Y, YY = sklearn.model_selection.train_test_split(X, Y, train_size=0.8, random_state=42, stratify=Y)
    reg = myRegression(sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=7, verbose=0, n_jobs=-1, class_weight='balanced'),
                        num_class)
    reg.fit(X, Y)
    pred = reg.predict_proba(XX)
    pred = pred[YY.reshape(-1)]
    print("           <Debug Info>        train:")
    reg.score(X, Y)
    print("           <Debug Info>        test:")
    reg.score(XX, YY)
    true_indicator = keras.utils.to_categorical(YY, num_classes=num_class)
    return sklearn.metrics.log_loss(true_indicator, pred)/math.log(num_class)

################################# Init For Root Node #################################
# init kmeans centroid with center of samples from each label, then do kmeans
def Init_By_Class(X, Y, num_class, sep_num, trial):
    init_centroid = []
    for i in range(np.unique(Y).shape[0]):
        Y = Y.reshape(-1)
        init_centroid.append(np.mean(X[Y==i],axis=0).reshape(-1))
    tmpH = 10
    init_centroid = np.array(init_centroid)
    if sep_num == num_class:
        trial = 1
    for i in range(trial):
        t = np.arange(0, np.unique(Y).shape[0]).tolist()
        tmp_idx = np.array(random.sample(t, len(t)))
        km = KMeans(n_clusters=sep_num, n_jobs=10, init=init_centroid[tmp_idx[0:sep_num]]).fit(X)
        ce = Comupte_Cross_Entropy(X[km.labels_ == i], Y[km.labels_ == i], num_class)
        if tmpH > ce:
            kmeans = km
            tmpH = ce
    data = []
    data.append({'Data': [],
                'Label': [],
                'Centroid': np.mean(X, axis=1),
                'H': Comupte_Cross_Entropy(X, Y, num_class),
                'ID': str(-1)})
    H = []
    Hidx = [1]
    for i in range(sep_num):
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
    Hidx = [0, 1]
    return data, H, Hidx

################################# KMeans Init Methods #################################
# LBG initialization
def Init_LBG(X, sep_num):
    c1 = np.mean(X, axis=0).reshape(1,-1)
    st = np.std(X, axis=0)
    dic = {}
    new_centroid = c1
    for i in range(sep_num-1):
        n = np.random.randint(2, size=X.shape[1])
        n[n==0] = -1
        if str(n) in dic:
            continue  
        dic[str(n)] = 1 
        c2 = c1 + n * st
        new_centroid = np.concatenate((new_centroid, c2.reshape(1,-1)), axis=0)
    return new_centroid

################################# Ada_KMeans train #################################
# try multiply times when spliting the leaf node
def Multi_Trial(X, sep_num, batch_size, trial, num_class, err):
    init = ['k-means++', 'random', 'k-means++', 'random', 'k-means++', 'random', Init_LBG(X['Data'], sep_num)]
    H = X['H'] - err
    center = []
    t_entropy = np.zeros((trial+1))
    t_entropy[-1] = H + err
    for i in range(trial):
        if batch_size == None:
            kmeans = KMeans(n_clusters=sep_num, n_jobs=20, init=init[i%len(init)]).fit(X['Data'])
        else:
            kmeans = MiniBatchKMeans(n_clusters=sep_num, batch_size=batch_size).fit(X['Data'])
        weight = Compute_Weight(kmeans.labels_)
        for k in range(sep_num):
            t_entropy[i] += weight[k]*Comupte_Cross_Entropy(X['Data'][kmeans.labels_ == k], X['Label'][kmeans.labels_ == k], num_class)
        if t_entropy[i] < H:
            H = t_entropy[i]
            center = kmeans.cluster_centers_.copy()
            label = kmeans.labels_.copy()
            print("           <Info>        Multi_Trial %s: Found a separation better than original! CE: %s"%(str(i),str(H)))
    print("           <Debug Info>        Gloabal entropy of each trail: %s"%(str(t_entropy)))
    if len(center) == 0:
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

def Ada_KMeans_train(X, Y, sep_num, trial, batch_size, minS, maxN, err, mvth, maxdepth, alpha):
    print("------------------- Start: Ada_KMeans_train")
    t0 = time.time()
    print("       <Info>        Trial: %s"%str(trial))
    print("       <Info>        Batch size: %s"%str(batch_size))
    print("       <Info>        Minimum number of samples in each cluster: %s"%str(minS))
    print("       <Info>        Max number of leaf nodes: %s"%str(batch_size))
    print("       <Info>        Stop splitting: %s"%str(err))
    print("       <Info>        Max depth: %s"%str(maxdepth))
    print("       <Info>        Alpha: %s"%str(alpha))
    # H: <list> entropy of nodes can be split
    # Hidx: <list> location of corresponding H in data
    num_class = np.unique(Y).shape[0]
    data, H, Hidx = Init_By_Class(X, Y, num_class, sep_num, trial)
    rootSampNum = Y.shape[0]
    global_H = []

    X, Y = [], []
    N ,myiter = 1, 1
    print("\n       <Info>        Start iteration")
    print("       <Info>        Iter %s"%(str(0)))
    global_H.append(Compute_GlobalH(data, rootSampNum, Hidx))
    while N < maxN:
        print("       <Info>        Iter %s"%(str(myiter)))
        idx = Select_Next_Split(data, Hidx, alpha)
        # finish splitting, when no node need further split 
        if H[idx] <= 0:
            print("       <Info>        Finish splitting!")
            break
        # if this cluster has too few sample, do not split this node
        if data[Hidx[idx]]['Data'].shape[0] < minS: 
            print("       <Warning>        Iter %s: Too small! continue for the next largest!"%str(myiter))
            H[idx] = -H[idx]
            continue
        # maxdepth
        if len(data[Hidx[idx]]['ID']) >= maxdepth: 
            print("       <Warning>        Depth >= maxdepth %s: Too small! continue for the next largest!"%str(maxdepth))
            H[idx] = -H[idx]
            continue
        # majority vote
        tmp = Majority_Vote(data[Hidx[idx]]['Label'], mvth)
        if tmp != -1:
            print("       <Warning>        Majority vote on this node, no further split needed!")
            H[idx] = -H[idx]
            data[Hidx[idx]]['Label'] = tmp * np.ones((data[Hidx[idx]]['Label'].shape[0]))
            continue 
        # try to split this node multi times
        subX = Multi_Trial(data[Hidx[idx]], sep_num=sep_num, batch_size=batch_size, trial=trial, num_class=num_class, err=err)
        # find a better splitting?
        if len(subX) != 0:
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
            global_H.append(Compute_GlobalH(data, rootSampNum, Hidx))
        else:
            print("       <Warning>        Iter %s: Don't split! continue for the next largest!"%str(myiter))
            H[idx] = -H[idx]
    data = Leaf_Node_Regression(data, Hidx, num_class)
    print("------------------- End: Ada_KMeans_train -> using %10f seconds"%(time.time()-t0))
    return data, global_H        

################################# Ada KMeans Test #################################
# list to dictionary
def List2Dict(data):
    res = {}
    for i in range(len(data)):
        res[data[i]['ID']] = data[i]
    return res

def Ada_KMeans_Iter_test(X, key_parent, data, sep_num):
    centroid = []
    key_child = []
    for i in range(sep_num):
        if key_parent+str(i) in data:
            centroid.append(data[key_parent+str(i)]['Centroid'].reshape(-1))
            key_child.append(key_parent+str(i))
    centroid = np.array(centroid)
    dist = sklearn.metrics.pairwise.euclidean_distances(X.reshape(1,-1), centroid).squeeze()
    key = key_child[np.argmin(dist)]
    if 'Regressor' in data[key]:
        return key   
    return Ada_KMeans_Iter_test(X, key, data, sep_num)

def Ada_KMeans_test(X, data, sep_num):
    for i in range(X.shape[0]):
        if i == 0:
            pred = data[Ada_KMeans_Iter_test(X[i], '', data, sep_num)]['Regressor'].predict_proba(X[i].reshape(1,-1))
        else:
            pred = np.concatenate((pred, data[Ada_KMeans_Iter_test(X[i], '', data, sep_num)]['Regressor'].predict_proba(X[i].reshape(1,-1))), axis=0)
    return pred

################################# MAIN Function #################################
def Ada_KMeans(X, Y=None, path='tmp.pkl', train=True, sep_num=2, trial=6, batch_size=10000, minS=300, maxN=50, err=0.005, mvth=0.99, maxdepth=50, alpha=1):
    print("=========== Start: Ada_KMeans")
    print("       <Info>        Input shape: %s"%str(X.shape))
    print("       <Info>        train: %s"%str(train))
    t0 = time.time()
    if train == True:
        data, globalH = Ada_KMeans_train(X, Y, sep_num=sep_num, trial=trial, batch_size=batch_size, minS=minS, maxN=maxN, err=err, mvth=mvth, maxdepth=maxdepth, alpha=alpha)
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

################################# Test #################################
if __name__ == "__main__":
    import cv2
    X = cv2.imread('../../data/test.jpg')
    X = cv2.resize(X, (40,40))
    X = X.reshape(-1,3)
    Y = np.random.randint(3, size=X.shape[0])
    print(" \n> This is a test enample: ")
    Y = Y.reshape(-1,1)
    #print(" \n--> Input X... \n", X)
    #print(" \n--> Input Y... \n", Y)
    data = Ada_KMeans(X, Y, err=1, maxN=10)
    print(" \n--> Result centroids... \n", data)
    