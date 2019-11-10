# v2019.11.08
# Alex
#
# use kmeans to separate bin
#
#   input:
#       x     -> (n, d)
#       y     -> (n,1)
#   return
#             -> number   

import numpy as np 
import math
from sklearn.cluster import MiniBatchKMeans, KMeans

class Cross_Entropy1D():
    def __init__(self, num_class, num_bin=10, batch_size=10):
        self.num_class = num_class
        self.num_bin = num_bin
        self.batch_size = batch_size

    def bin_process(self, x ,y):
        mybin = np.zeros((self.num_bin, self.num_class))
        gst = np.zeros((self.num_class))
        if self.batch_size == None:
            kmeans = KMeans(n_clusters=self.num_bin, verbose=0, random_state=9).fit(x)
        else:
            kmeans = MiniBatchKMeans(n_clusters=self.num_bin, verbose=0, batch_size=self.batch_size).fit(x)
        label = kmeans.labels_
        for l in range(0, self.num_bin):
            st = y[label == l]
            for i in range(0, self.num_class):
                idx = (st == i)
                mybin[l, i] = st[idx].shape[0]
        for i in range(0, self.num_class):
            idx = (y == i)
            gst[i] = y[idx].shape[0]
        mybin /= gst
        return np.argmax(mybin, axis=1)

    def compute_prob(self, x, y):
        prob = np.zeros((self.num_class))
        mybin = self.bin_process(x, y)
        for l in range(0, self.num_class):
            p = mybin[mybin == l]
            p = np.array(p).shape[0]
            prob[l] = p / (float)(self.num_bin)
        return prob

    def compute(self, x, y, class_weight=None):
        x = x.astype('float64')
        y = y.astype('int64')
        prob = self.compute_prob(x, y)
        prob = -1 * np.log10(prob + 1e-5) / np.log10(self.num_class)
        y = np.moveaxis(y, 0, 1)
        H = 0
        for c in range(0, self.num_class):
            yy = (y == c)
            p = prob[c]
            H += np.mean(yy * p, axis=1)
        if class_weight != None:
            class_weight = np.array(class_weight)
            H *= class_weight.reshape(class_weight.shape[0],1) * self.num_class
        return np.sum(H, axis=0)

if __name__ == "__main__":
    import time
    t0 = time.time()
    x = np.array([1,2,3,1,3,5,7,1,2,4])
    y = np.array([0,0,1,0,1,0,1,0,1,1])
    ce = Cross_Entropy1D(2)
    H = ce.compute(x.reshape(10,1), y.reshape(10,1), class_weight=None)
    print(H)
    print('ideal: ', str(1.12576938))
    print('Using time: ', time.time()-t0)