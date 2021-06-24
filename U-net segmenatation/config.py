# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:45:25 2021

@author: Admin
"""
config = {
    
    "iteration"      : 200000,      # iterataions
    "learning_rate"  : 10**-3,     # learning rate
    "decay_lr"       : 0.5,        # deacy learing rate factor
    "decay_lr_iter"  : 20000,        # deacy learning rate iterataion
    "min_lr"         : 10**-5,     # min learing
    "batch_size"     : 32,         # batch size
    "optimizer_flag" : 'Adam',     # Optimizer
    
    "number_classes" : 3,          # number of classes
    
    "train points"   : 300,        # train data points
    
    "threshold loss" : 10,         # threshold loss
    
    
    "transform"      : [False, False, True], # flip, rnd crop, gaussian noise
    
    "ema_factor"     : 0.95,
    "consis_coef"    : 0.3,
    "alpha"          : 0.1,
    
    "warmup"         : 20000,
    
    "test_model_cycel" :250,
}