# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:24:38 2021

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
#from config import config

def plot(train_ious,test_ious):
    x1 = np.arange(len(train_ious))
    x2 = np.arange(len(test_ious))
    plt.plot(x1,train_ious,label='Train loss', c='r')
    plt.plot(x2,test_ious,label='Test loss', c='b')
    #plt.axhline(config["threshold loss"],0,len(Loss_train),label='loss threshold',c='k')
    plt.xlabel('Epoch')
    plt.ylabel('IOU')
    plt.legend()
    plt.show()