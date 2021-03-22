# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:55:48 2021

@author: Admin
"""
import numpy as np
from config import config

def load_data():
    
    data   = np.load(config["data_path"])/255
    labels = np.load(config["labels_path"])


    
    arr = np.arange(data.shape[0])
    np.random.shuffle(arr)
     
    data   = data[arr]
    labels = labels[arr]
    
    images_train = data[:config["train points"]]
    labels_train = labels[:config["train points"]]
    images_test  = data[config["train points"]:]
    labels_test  = labels[config["train points"]:]
    
    np.save('data/images_train.npy', images_train)
    np.save('data/labels_train.npy', labels_train)
    
    np.save('data/images_test.npy', images_test)
    np.save('data/labels_test.npy', labels_test)
    
    return images_train, labels_train,images_test,labels_test