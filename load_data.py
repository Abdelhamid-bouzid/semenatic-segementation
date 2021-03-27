# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:55:48 2021

@author: Admin
"""
import numpy as np

def load_data():
    
    images_train = np.load('data/images_train.npy')/255
    labels_train = np.load('data/labels_train.npy')
    
    images_test  = np.load('data/images_test.npy')/255
    labels_test  = np.load('data/labels_test.npy')
    
    return images_train, labels_train,images_test,labels_test