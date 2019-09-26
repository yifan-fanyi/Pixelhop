import numpy as np 
import math
import cv2
import os
import time
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from progress.bar import Bar
from skimage.measure import block_reduce


def normailze(x):
    x.astype('float32')
    return 255.*(x - np.min(x))/(np.max(x)-np.min(x)+0.000000001)

def myRandomForest(X, Y, n_estimators=10, saved_name='myRF_'+str(time.time())+'.pkl'):
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=None, verbose=1, n_jobs=10)
    model.fit(X, Y)
    importances = model.feature_importances_
    print("       <Info>        feature importance",importances)
    joblib.dump(model, '../model/'+saved_name)
    print("       <Info>        Save RF regressor model as name: %10s"%(saved_name))
    return model

def myRandomForestC(X, Y, n_estimators=10, saved_name='myRF_'+str(time.time())+'.pkl'):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, verbose=1, n_jobs=10)
    model.fit(X, Y)
    importances = model.feature_importances_
    print("       <Info>        feature importance",importances)
    joblib.dump(model, '../model/'+saved_name)
    print("       <Info>        Save RF regressor model as name: %10s"%(saved_name))
    return model

def getTrainXY_ref(X, Y, img, ignore_dist=2, th1=100, th2=400, keep_balance=True, keep_rate=0.99):
    print("=========== Start: getTrainXY_ref")
    t0 = time.time()
    #(n, h, w, d)
    #(n, h, w)
    new_X0 = []
    new_Y0 = []
    new_X1 = []
    new_Y1 = []
    S = X.shape
    for n in range(0,S[0]):
        ref = cv2.Canny(img[n], th1, th2)   
        for i in range(ignore_dist,S[1]-ignore_dist):
            for j in range(ignore_dist,S[2]-ignore_dist):
                f = ref[i,j]
                if f < 50:
                    continue
                if np.sum(Y[n,i-0:i+1,j-1:j+1,0]) > 50:
                #if Y[n,i,j,0] > 100:
                    new_X0.append(X[n,i,j])
                    new_Y0.append(1)
                    #X[n,i,j] = np.array([255,0,255])
                elif Y[n,i,j,0] > 0 or np.sum(Y[n,i-ignore_dist:i+ignore_dist,j-ignore_dist:j+ignore_dist,0]) > 50:
                    continue
                else:
                    new_X1.append(X[n,i,j])
                    new_Y1.append(0)
                    #X[n,i,j] = np.array([0,255,255])
    new_X0 = np.array(new_X0).reshape(-1,S[3])
    new_X1 = np.array(new_X1).reshape(-1,S[3])
    new_Y0 = np.array(new_Y0).reshape(-1)
    new_Y1 = np.array(new_Y1).reshape(-1)
    print("       <Info>        getTrainXY: count for postive label: %10d, count for negative label: %10d"%(new_Y1.shape[0],new_Y0.shape[0]))
    if keep_balance == True:
        n = (int)(keep_rate*min(new_Y0.shape[0], new_Y1.shape[0]))
        new_X0, X_test, new_Y0, y_test = train_test_split(new_X0, new_Y0, train_size=n, random_state=42)
        new_X1, X_test, new_Y1, y_test = train_test_split(new_X1, new_Y1, train_size=n, random_state=42)
    new_X = np.concatenate((new_X0, new_X1),axis=0)
    new_Y = np.concatenate((new_Y0, new_Y1),axis=0)
    #cv2.imwrite('./tmp/x.jpg',X[0])
    print("=========== End: getTrainXY_ref -> using %10f seconds"%(time.time()-t0))
    return new_X, new_Y

def getTrainXY(X, Y, ignore_dist=5, keep_balance=True, keep_rate=0.99):
    print("=========== Start: getTrainXY")
    t0 = time.time()
    new_X0 = []
    new_Y0 = []
    new_X1 = []
    new_Y1 = []
    S = X.shape
    for n in range(0,S[0]):
        for i in range(ignore_dist,S[1]-ignore_dist):
            for j in range(ignore_dist,S[2]-ignore_dist):
                if np.sum(Y[n,i,j,0]) > 50:
                #if np.sum(Y[n,i-1:i+1,j-1:j+1,0]) > 50:
                    new_X0.append(X[n,i,j])
                    new_Y0.append(1)
                    #X[n,i,j] = np.array([255,0,255])
                elif Y[n,i,j,0] > 0 or np.sum(Y[n,i-ignore_dist:i+ignore_dist,j-ignore_dist:j+ignore_dist,0]) > 50:
                    continue
                else:
                    new_X1.append(X[n,i,j])
                    new_Y1.append(0)
                    #X[n,i,j] = np.array([0,255,255])
    new_X0 = np.array(new_X0).reshape(-1,S[3])
    new_X1 = np.array(new_X1).reshape(-1,S[3])
    new_Y0 = np.array(new_Y0).reshape(-1)
    new_Y1 = np.array(new_Y1).reshape(-1)
    print("       <Info>        getTrainXY: count for postive label: %10d, count for negative label: %10d"%(new_Y1.shape[0],new_Y0.shape[0]))
    if keep_balance == True:
        n = (int)(keep_rate*min(new_Y0.shape[0], new_Y1.shape[0]))
        new_X0, X_test, new_Y0, y_test = train_test_split(new_X0, new_Y0, train_size=n, random_state=42)
        new_X1, X_test, new_Y1, y_test = train_test_split(new_X1, new_Y1, train_size=n, random_state=42)
    new_X = np.concatenate((new_X0, new_X1),axis=0)
    new_Y = np.concatenate((new_Y0, new_Y1),axis=0)
    print("=========== End: getTrainXY_ref1 -> using %10f seconds"%(time.time()-t0))
    return new_X, new_Y