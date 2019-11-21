# LAG example
# 2019.11.09

import numpy as np 
from skimage.measure import block_reduce
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from framework.old.data import *
from framework.old.LAG import *

from framework.layer import *
from framework.pixelhop import *

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels, class_list = import_data_mnist("0-9")  
    train_images = train_images[:1000]
    train_labels = train_labels[:1000]
    test_images = train_images[:500]
    test_labels = train_labels[:500]
    
    train_feature = PixelHop_Unit(train_images, dilate=np.array([1]), pad='reflect', num_AC_kernels=5, weight_name='pixelhop1_mnist.pkl', getK=1, batch=None)
    train_feature = block_reduce(train_feature, (1, 4, 4, 1), np.mean).reshape(1000,-1)
    train_feature_reduce = LAG_Unit(train_feature, train_labels, class_list=class_list, num_clusters=[5,5,5,5,5,5,5,5,5,5], alpha=5, train=True)
    
    test_feature = PixelHop_Unit(test_images, dilate=np.array([1]), pad='reflect', num_AC_kernels=5, weight_name='pixelhop1_mnist.pkl', getK=0, batch=None)
    test_feature = block_reduce(test_feature, (1, 4, 4, 1), np.mean).reshape(500,-1)
    test_feature_reduce = LAG_Unit(test_feature, class_list=class_list, num_clusters=[5,5,5,5,5,5,5,5,5,5],alpha=5,train=False)
    
    scaler = preprocessing.StandardScaler()
    feature = scaler.fit_transform(train_feature_reduce)
    feature_test = scaler.transform(test_feature_reduce)     
   
    clf=SVC().fit(feature, train_labels) 
    print('***** Train ACC:', accuracy_score(train_labels,clf.predict(feature)))
    print('***** Test ACC:', accuracy_score(test_labels,clf.predict(feature_test)))