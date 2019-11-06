# v2019.11.06
# triplet loss regression
# input :
#   X           -> Anchor feature : Positive feature : Negative feature
#   new_dim     -> Embedding dimension

import os
import pickle
import h5py
import numpy as np
import keras
from keras import backend as K
from keras.layers import Dense

batch_size = 1
a1 = 0.6

def triplet_loss(y_true, y_pred):
    y_pred = K.l2_normalize(y_pred,axis=1)
    batch = batch_size
    ref1 = y_pred[0:batch,:]
    pos1 = y_pred[batch:batch+batch,:]
    neg1 = y_pred[batch+batch:3*batch,:]
    dis_pos = K.sum(K.square(ref1 - pos1), axis=1, keepdims=True)
    dis_neg = K.sum(K.square(ref1 - neg1), axis=1, keepdims=True)
    dis_pos = K.sqrt(dis_pos)
    dis_neg = K.sqrt(dis_neg)
    d1 = K.maximum(0.0, dis_pos - dis_neg + a1)
    return K.mean(d1)

def Triplet_model(new_dim, input_shape, optimizer):
    model = keras.Sequential()
    model.add(Dense(units=new_dim, input_shape=input_shape, name='layer1'))
    model.compile(optimizer=optimizer,loss=triplet_loss)
    return model

def Triplet_train(X, saved_name, new_dim, optimizer):
    model = Triplet_model(new_dim, (X.shape[1],), optimizer)
    model.fit(X, np.ones([batch_size*3, new_dim]), verbose=0)   
    model.save_weights('../weight/'+saved_name)

def Triplet_test(X, saved_name, new_dim, optimizer):
    model = Triplet_model(new_dim, (X.shape[1],), optimizer)
    model.load_weights('../weight/'+saved_name, by_name=True)
    return model.predict(X)

def Triplet_unit(X, new_dim=10, a=0.6, train=True, saved_name='Triplet.h5', optimizer='sgd'):
    global batch_size, a1
    batch_size = X.shape[0] // 3
    a1 = a
    if train == True:
        Triplet_train(X, saved_name, new_dim, optimizer)
    X = Triplet_test(X, saved_name, new_dim, optimizer)
    return X

if __name__ == "__main__":
    print(" \n> This is a test enample: ")
    X = np.array([[-1, -1, 1], [-1, -2, 1], [-2, -1, 1], [-2, -2, 1], [1, 1, 5], [2, 3, 4]])
    print(" \n--> Input... \n", X)
    X = Triplet_unit(X, new_dim=5, train=1)
    print(" \n--> Embedding... \n",X)