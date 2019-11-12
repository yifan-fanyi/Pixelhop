# v2019.10.27.2
# <2019.11.12> old cross entropy, method of compute cross entropy has be changed
# Alex
#
# Reference:
#       Manimaran A, Ramanathan T, You S, et al. Visualization, Discriminability and Applications of Interpretable Saak Features[J]. 2019.
# Compute cross entropy across each feature for feature selection
#   input:
#       x     -> (n, d)
#       y     -> (n,1)
#   return
#             -> (d)   

import numpy as np 
import math

class Cross_Entropy():
    def __init__(self, num_class, num_bin=10):
        self.num_class = num_class
        self.num_bin = num_bin

    def bin_process(self, x ,y):
        x = ((x - np.min(x)) / (np.max(x) - np.min(x))) * (self.num_bin)
        mybin = np.zeros((self.num_bin, self.num_class))
        b = x.astype('int64')
        b[b == self.num_bin] -= 1
        mybin[b,y] += 1.
        for l in range(0,self.num_class):
            p = np.array(y[ y==l ]).shape[0]
            mybin[:,l] /= (float)(p)
        return np.argmax(mybin, axis=1)

    def compute_prob(self, x, y):
        prob = np.zeros((self.num_class, x.shape[1]))
        for k in range(0, x.shape[1]):
            mybin = self.bin_process(x[:,k], y[:,0])
            for l in range(0, self.num_class):
                p = mybin[mybin == l]
                p = np.array(p).shape[0]
                prob[l,k] = p / (float)(self.num_bin)
        return prob

    def compute(self, x, y, class_weight=None):
        x = x.astype('float64')
        y = y.astype('int64')
        prob = self.compute_prob(x, y)
        prob = -1 * np.log10(prob + 1e-5) / np.log10(self.num_class)
        y = np.moveaxis(y, 0, 1)
        H = np.zeros((self.num_class, x.shape[1]))
        for c in range(0, self.num_class):
            yy = y == c
            p = prob[c].reshape(prob.shape[1], 1)
            p = p.repeat(yy.shape[1], axis=1)
            H[c] += np.mean(yy * p, axis=1)
        if class_weight != None:
            class_weight = np.array(class_weight)
            H *= class_weight.reshape(class_weight.shape[0],1) * self.num_class
        return np.sum(H, axis=0)

if __name__ == "__main__":
    import time
    t0 = time.time()
    x = np.array([1,2,3,1,3,5,7,1,2,4])
    y = np.array([0,0,1,0,1,0,1,0,1,1])
    ce = Cross_Entropy(2)
    H = ce.compute(x.reshape(10,1), y.reshape(10,1), class_weight=None)
    print(H)
    print('ideal: ', str([1.12576938]))
    print('Using time: ', time.time()-t0)