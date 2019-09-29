#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:36:02 2019

@author: yueru
"""


from keras.datasets import cifar10,mnist,fashion_mnist
import numpy as np
from scipy import ndimage
from skimage.color import rgb2lab

def get_data_for_class(images, labels, cls):
	if type(cls)==list:
		idx=np.zeros(labels.shape, dtype=bool)
		for c in cls:
			idx=np.logical_or(idx, labels==c)
	else:
		idx=(labels==cls)
	return images[idx], labels[idx]
def parse_list_stringe(list_string):
    """Convert the class string to list."""
    elem_groups = list_string.split(",")
    results = []
    for group in elem_groups:
        term = group.split("-")
        if len(term) == 1:
            if float(term[0]) >= 1.0:
                results.append(int(term[0]))
            else:
                results.append(float(term[0]))
        else:
            if float(term[0]) >= 1.0:
                start = int(term[0])
                end = int(term[1])
                results += range(start, end + 1)
            else:
                start = float(term[0])
                end = float(term[1])
                results += range(start, end + 1)
    return results
def import_data(use_classes):
	(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
	train_images=train_images/255.
	test_images=test_images/255.
	class_list=[0,1,2,3,4,5,6,7,8,9]
		
	return train_images, train_labels, test_images, test_labels, class_list
def import_data_fashion_mnist(use_classes):
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	train_images=train_images.reshape(-1,28,28,1)
	test_images=test_images.reshape(-1,28,28,1)
	train_images=train_images/255.
	test_images=test_images/255.
	# print(train_images.shape) # 60000*28*28*1

	# zeropadding
	train_images=np.pad(train_images, ((0,0),(2,2),(2,2),(0,0)), mode='constant')
	test_images=np.pad(test_images,  ((0,0),(2,2),(2,2),(0,0)), mode='constant')
	# print(train_images.shape) # 60000*32*32*1

	if use_classes!='0-9':
		class_list=parse_list_string(use_classes)
		train_images, train_labels=get_data_for_class(train_images, train_labels, class_list)
		test_images, test_labels=get_data_for_class(test_images, test_labels, class_list)
		# print(class_list)
	else:
		class_list=[0,1,2,3,4,5,6,7,8,9]
		
	return train_images, train_labels, test_images, test_labels, class_list
def import_data_mnist(use_classes):
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images=train_images.reshape(-1,28,28,1)
	test_images=test_images.reshape(-1,28,28,1)
	train_images=train_images/255.
	test_images=test_images/255.
	# print(train_images.shape) # 60000*28*28*1

	# zeropadding
	train_images=np.pad(train_images, ((0,0),(2,2),(2,2),(0,0)), mode='constant')
	test_images=np.pad(test_images,  ((0,0),(2,2),(2,2),(0,0)), mode='constant')
	# print(train_images.shape) # 60000*32*32*1

	if use_classes!='0-9':
		class_list=parse_list_string(use_classes)
		train_images, train_labels=get_data_for_class(train_images, train_labels, class_list)
		test_images, test_labels=get_data_for_class(test_images, test_labels, class_list)
		# print(class_list)
	else:
		class_list=[0,1,2,3,4,5,6,7,8,9]
		
	return train_images, train_labels, test_images, test_labels, class_list

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)
def convert_image(data,format):
#    if data.shape[1] < 10:
#        data=np.moveaxis(data, 3,1)
    if format == 'YCbCr' or format == 'LAB':
        new_data = np.zeros((data.shape))
    else:
        new_data = np.zeros((data.shape[0],data.shape[1],data.shape[2]))
    for i in range(data.shape[0]):
        im_ycbcr = rgb2ycbcr(data[i]*255.)/255.
        if format == 'YCbCr':
            new_data[i] = im_ycbcr
        if format == 'LAB':
            new_data[i] = Normalize(rgb2lab(data[i]), ABS=False)
        if format == 'L':
            new_data[i] = Normalize(rgb2lab(data[i])[:,:,0], ABS=False)
        if format == 'A':
            new_data[i] = Normalize(rgb2lab(data[i])[:,:,1], ABS=False)
        if format == 'B':
            new_data[i] = Normalize(rgb2lab(data[i])[:,:,2], ABS=False)            
        if format == 'R':
            new_data[i] = data[i,:,:,0]
        elif format == 'G':
            new_data[i] = data[i,:,:,1]
        elif format == 'B':
            new_data[i] = data[i,:,:,2]
        elif format == 'Y':
            new_data[i] = im_ycbcr[:,:,0]
        elif format == 'Cb':
            new_data[i] = Normalize(im_ycbcr[:,:,1] , ABS=False)           
        elif format == 'Cr':
            new_data[i] = Normalize(im_ycbcr[:,:,2] , ABS=False)  
        elif format == 'Gray':
            new_data[i] = 0.2989 * data[i,:,:,0] + 0.5870 * data[i,:,:,1] + 0.1140 * data[i,:,:,2]
    if format == 'YCbCr' or format == 'LAB':   
        return new_data
    else:
        return new_data.reshape(data.shape[0],data.shape[1],data.shape[2],1)
def Normalize(data, ABS=True):
    if ABS:
        data_new = np.abs(data)
    else:
        data_new = data
    if len(data_new.shape)==2:
        data_new = (data_new - data_new.min())
        data_new = data_new/(data_new.max()+1e-5)
    else:
        data_new = data_new - np.min(data_new.reshape(-1,3),0).reshape(1,1,-1)
#        print np.min(data_new.reshape(-1,3),0)
        data_new = data_new/(np.max(data_new.reshape(-1,3),0).reshape(1,1,-1)+1e-5)      
    return data_new
   
def transform_data(data,TYPE):
#    print 'transform data:', TYPE    
    L3 = np.array([1, 2, 1]).reshape(3,1)
    S3 = np.array([1, -2, 1]).reshape(3,1)
    E3 = np.array([1, 0, -1]).reshape(3,1)
    L3L3 = np.dot(L3,L3.transpose())
    E3E3 = np.dot(E3,E3.transpose())
    S3S3 = np.dot(S3,S3.transpose())
    L3E3 = np.dot(L3,E3.transpose())
    L3S3 = np.dot(L3,S3.transpose())
    E3L3 = np.dot(E3,L3.transpose())
    E3S3 = np.dot(E3,S3.transpose())
    S3L3 = np.dot(S3,L3.transpose())
    S3E3 = np.dot(S3,E3.transpose())
#    kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
#    ky = np.array([[1,1,1] ,[0,0,0], [-1,-1,-1]])
    kxy = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
    kyx = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
    transformed_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        img = data[i,:,:,0]
        # Get x-gradient in "sx"
        if TYPE == 'sx':
            transformed_data[i,:,:,0] = Normalize(ndimage.sobel(img,axis=0,mode='reflect'))
        # Get y-gradient in "sy"
        if TYPE == 'sy':
            transformed_data[i,:,:,0] = Normalize(ndimage.sobel(img,axis=1,mode='reflect'))
        if TYPE == 'xy':
            transformed_data[i,:,:,0] = Normalize(ndimage.convolve(img,kxy,mode='reflect'))
        # Get y-gradient in "sy"
        if TYPE == 'yx':
            transformed_data[i,:,:,0] = Normalize(ndimage.convolve(img,kyx,mode='reflect'))
        if TYPE == 'sxy':
            sx = ndimage.sobel(img,axis=0,mode='reflect')
            sy = ndimage.sobel(img,axis=1,mode='reflect')
        # Get square root of sum of squares
            transformed_data[i,:,:,0]=np.hypot(sx,sy)
        # Hopefully see some edges
        if TYPE == 'lp':
            transformed_data[i,:,:,0]=Normalize(ndimage.laplace(img,mode='reflect'))
        # LAWS
        if TYPE == 'l3l3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,L3L3,mode='reflect'))
        if TYPE == 'e3e3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,E3E3,mode='reflect'))
        if TYPE == 's3s3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,S3S3,mode='reflect'))
        if TYPE == 'l3e3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,L3E3,mode='reflect'))
        if TYPE == 'l3s3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,L3S3,mode='reflect'))
        if TYPE == 'e3l3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,E3L3,mode='reflect'))
        if TYPE == 'e3s3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,E3S3,mode='reflect'))
        if TYPE == 's3l3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,S3L3,mode='reflect'))
        if TYPE == 's3e3':
            transformed_data[i,:,:,0]=Normalize(ndimage.convolve(img,S3E3,mode='reflect'))
    return transformed_data