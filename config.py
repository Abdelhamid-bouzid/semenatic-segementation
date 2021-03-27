# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:45:25 2021

@author: Admin
"""
config = {
    
    "Epochs"         : 300,       # Number of epochs
    "learning_rate"  : 10**-3,     # learning rate
    "decay_lr"       : 0.1,        # deacy learing
    "decay_lr_epoch" : 300,         # deacy learning rate
    "min_lr"         : 10**-5,     # min learing
    "batch_size"     : 8,         # batch size
    "optimizer_flag" : 'Adam',     # Optimizer
    
    "train points"   : 200,        # train data points
    
    "threshold loss" : 10,         # threshold loss
    
    
    "data_path"      : 'data/data.npy',
    "labels_path"    : 'data/labels.npy',
}