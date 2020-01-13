# v2020.01.13
import numpy as np 
from numpy import linalg
import sklearn
import pickle
from  sklearn.decomposition import PCA

# par: {'0': {'PCA':xx, 'idx': xx, 'Energy':xx, 'Bias':xx}}
def Single_PCA_train(X, Y, num_feature_piter=1):
    X_mean = np.mean(X, axis=0, keepdims=True)
    par = {'idx': np.zeros((num_feature_piter), dtype='int'), 'Energy': np.zeros((num_feature_piter))}
    par['PCA'] = PCA(n_components=min(X.shape[1],X.shape[0]), svd_solver='full').fit(X) 
    X_transform = par['PCA'].transform(X) 
    for i in range(num_feature_piter):
        par['idx'][i] = i
        par['Energy'][i] = par['PCA'].explained_variance_ratio_[par['idx'][i]]
    par['Bias'] = np.max(linalg.norm(X_transform[:,par['idx']], axis=1))
    fea = X_transform[:,par['idx']] + par['Bias']
    X_transform[:,par['idx']] = 0
    X_mean[:, par['idx']] = 0
    X = np.dot(X_transform, par['PCA'].components_) + X_mean
    return X, fea, par 

def Single_PCA_test(X, par):
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_transform = par['PCA'].transform(X) 
    fea = X_transform[:,par['idx']] + par['Bias']
    X_transform[:,par['idx']] = 0
    X_mean[:, par['idx']] = 0
    X = np.dot(X_transform, par['PCA'].components_) + X_mean
    return X, fea

def rPCA_train(X, Y, num_feature_piter=1, energy=0.99, num_feature=None):
    par = {}
    it = 0
    eng = 0.0
    while 1:
        X, tmp, par[str(it)] = Single_PCA_train(X, Y, num_feature_piter)
        if it == 0:
            fea = tmp
        else:
            fea = np.concatenate((fea, tmp), axis=1)
        eng = eng + (1 - eng) * np.sum(par[str(it)]['Energy'])
        it += 1
        if (energy != None and eng > energy) or (num_feature != None and it == num_feature):
            break
    return fea, par

def rPCA_test(X, par):
    it = 0
    while 1:
        if str(it) not in par:
            break
        X, tmp = Single_PCA_test(X, par[str(it)])
        if it == 0:
            fea = tmp
        else:
            fea = np.concatenate((fea, tmp), axis=1)
        it += 1
    return fea

def rPCA(X, Y, num_feature_piter, energy, path, train, num_feature):
    if train == True:
        fea, par = rPCA_train(X, Y, num_feature_piter, energy, num_feature)
        f = open(path, 'wb')
        pickle.dump(par, f)
        f.close()
        return fea
    f = open(path, 'rb')
    par = pickle.load(f)
    f.close()
    fea = rPCA_test(X, par)
    return fea

if __name__ == "__main__":
    import cv2
    X = cv2.imread('3063.jpg').reshape(-1,3)
    X = X.astype('float64')
    Y = cv2.imread('3063t.jpg', 0).reshape(-1,1)
    Y[Y!=0] = 1
    rPCA_train(X[0:100], Y[0:100], num_feature=1, energy=None, num_feature=2)
    print(X[0:3])
    fea = rPCA(X[0:100], Y[0:100], num_feature_piter=1, energy=None, path='tmp.pkl', train=0, num_feature=2)
    print(fea[0:3])