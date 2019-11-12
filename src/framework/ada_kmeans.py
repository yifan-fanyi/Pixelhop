# v2019.11.05

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

def encode_soft(label, total=None):
    soft_labels = np.zeros(len(total))
    if label.size>1:
        label = list(label)
        for i in range(len(total)):
            c = label.count(i)
            soft_labels[i] = float(c)/total[i]
    else:
        soft_labels[label] = 1.0
    soft_labels = soft_labels/np.sum(soft_labels)
    return soft_labels

def Continue_split(H, limit):
    if H<limit:
        return False
    else:
        return True

def Comupte_Cross_Entropy(X, Y, total=None):
    cls_num = len(total)
    samp_num = Y.size
    true_label = np.copy(Y[:,0])    
    true_indicator = np.zeros((samp_num,cls_num))
    true_indicator[np.arange(samp_num),Y[:,0]] = 1    
    pred_label = np.zeros(Y.shape[0])
    pred_soft = np.zeros((true_label.shape[0],cls_num))
    if samp_num<30:
        pred_label = np.argmax(encode_soft(true_label,total=total))
        pred_soft = np.repeat(encode_soft(true_label,total=total).reshape(1,-1),samp_num,axis=0)
    else:
        samp = np.copy(X)
        K2 = 12
        kmeans = MiniBatchKMeans(n_clusters = K2, batch_size=min(samp_num,100000)).fit(samp)
        k_label2 = kmeans.predict(samp)
        for k2 in range(K2):
            idx2 = np.where(k_label2==k2)[0]
            label2 = true_label[idx2]
            pred_label[idx2] = np.argmax(encode_soft(label2,total=total))
            pred_soft[idx2] = encode_soft(label2,total=total)
    return LL(true_indicator,pred_soft)/math.log(cls_num)

def Comupte_Cross_Entropy1(X, Y):
    if np.unique(Y).shape[0] == 1: #alread pure
        return 0
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X, Y)
    prob = clf.predict_proba(X)
    logprob = np.log10(prob[Y])/np.log10(2)
    return -np.mean(logprob)

#only for 2
def Compute_Weight(Y):
    weight = np.zeros((2))
    for i in range(0,weight.shape[0]):
        weight[i] = 1-(float)(Y[Y==i].shape[0])/(float)(Y.shape[0])
    return weight

def Multi_Trial(X, sep_num=2, batch_size=None, trial=6, total=None):
    init = ['k-means++','random','k-means++','random','k-means++','random']
    H = X['H']
    center = []
    for i in range(trial):
        tH = 0.0
        if batch_size == None:
            kmeans = KMeans(n_clusters=sep_num,n_jobs=10,init = init[i]).fit(X['Data'])
        else:
            kmeans = MiniBatchKMeans(n_clusters=sep_num, batch_size=batch_size).fit(X['Data'])
        # early stop
        k_labels = kmeans.labels_
        counting = np.array(Counter(k_labels.tolist()).most_common(np.unique(k_labels).size))[:,1]
        if np.min(counting)>int(0.05*X['Data'].shape[0]):
            weight = Compute_Weight(kmeans.labels_)
            for k in range(sep_num):
                tH = tH + weight[k]*Comupte_Cross_Entropy1(X['Data'][kmeans.labels_ == k], X['Label'][kmeans.labels_ == k])
            #tH /= float(sep_num)
            if tH < H:
                H = np.copy(tH)
                center = kmeans.cluster_centers_
                label = kmeans.labels_
                print("            <Multi_Trial {}>    Found a separation better than original!".format(i))
                print("            <Multi_Trial {}>    New ce = {}".format(i,H))
    if len(center)==0:
        return []
    subX = []
    for i in range(sep_num):
        idx = (label == i)
        subX.append({'Data':X['Data'][idx], 'Label':X['Label'][idx], 'Centroid':center[i], 'H':Comupte_Cross_Entropy1(X['Data'][idx],X['Label'][idx]),'ID':X['ID']+str(i)})
    return subX 

def Ada_KMeans(X, Y, trial=6, batch_size=10000, minS=0.1, maxN=50, limit=0.5, maxiter=50):
    # trial: # of runs in each separation
    # minS: minimum percent of samples in each cluster
    # maxN: max number of leaf nodes (centroids)
    # limit: stop splitting when the max CE<limit
    # max iteration
    Num_sample = X.shape[0]
    data = [{'Data':X, 'Label':Y, 'Centroid':np.mean(X,axis=0), 'H':Comupte_Cross_Entropy1(X, Y),'ID':'0'}]
    X = []
    H = [data[0]['H']]
    Hidx = [0,1]
    N = 1
    myiter = 1
    
    print("        > Start!")
    while N < maxN and myiter < maxiter:
        #print("        > Ite--{}: {}".format(myiter,H))
        idx = np.argmax(np.array(H))
        if Continue_split(H[idx], limit) == False: # continue to split?
            print("        > Finish splitting!")
            break
        print(idx, Hidx, H)
        if data[Hidx[idx]]['Data'].shape[0] < int(minS*Num_sample): # if this cluster has too few sample, change the next largest
            print("        > Ite--{}: Too small! continue for the next largest".format(myiter))
            H[idx] = -2
            continue
        subX = Multi_Trial(data[Hidx[idx]], batch_size=batch_size, trial=trial)
        if len(subX)!=0:
            # save memory
            data[Hidx[idx]]['Data'] = []
            data[Hidx[idx]]['Label'] = []
            data += subX
            H.pop(idx)
            Hidx.pop(idx)
            for d in subX:
                H.append(d['H'])
                Hidx.append(Hidx[-1]+1)
                N += 1
            myiter += 1
        else:
            print("        > Ite--{}: Don't split! continue for the next largest".format(myiter))
            H[idx] = -1

    return Merge(data, Hidx)

def Merge(data, Hidx):
    centroid = []
    X = []
    Y = []
    cluster_label = []
    label = []
    for i in range(0, len(Hidx)-1):
        if data[Hidx[i]]['H'] == -2:
            continue
        X.append(data[Hidx[i]]['Data'])
        Y.append(data[Hidx[i]]['Label'])
        centroid.append(data[Hidx[i]]['Centroid'])
        t = data[Hidx[i]]['Label']
        if t[t == 0].shape[0] > t[t == 1].shape[0]:
            cluster_label.append(0)
        elif t[t == 0].shape[0] < t[t == 1].shape[0]:
            cluster_label.append(1)
        label.append(i*np.ones(data[Hidx[i]]['Label'].shape[0]))
    X = np.array(X)
    X = X.reshape(-1, X.shape[-1])
    Y = np.array(Y)
    centroid = np.array(centroid)
    cluster_label = np.array(cluster_label)
    label = np.array(label)
    return X, Y, label, centroid, cluster_label


if __name__ == "__main__":
    print(" \n> This is a test enample: ")
    X = np.array([[-1, -1, 1], [-1, -2, 1], [-2, -1, 1], [-2, -2, 1], [1, 1, 5], [2, 3, 4]])
    Y = np.array([0, 0, 0, 1, 1, 1])
    Y = Y.reshape(-1,1)
    print(" \n--> Input X... \n", X)
    print(" \n--> Input Y... \n", Y)
    data = Ada_KMeans(X, Y, limit=1, maxN=10)
    print(" \n--> Result centroids... \n", data)