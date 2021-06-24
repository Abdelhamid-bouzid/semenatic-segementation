
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:45:05 2021

@author: Admin
"""
import torch
from load_data import load_data
from learning_function import learning_function
from torchsummary import summary
from plot import plot
from Unet import UNet
from config import config


#####################################################################################################
######################################## load data ##################################################
#####################################################################################################
l_train = load_data(r"D:/MHS data segmentation labeling/data/l_train.npy")
test    = load_data(r"D:/MHS data segmentation labeling/data/test.npy")

#####################################################################################################
#################################### student model ##################################################
#####################################################################################################
model = UNet(config["number_classes"])
#summary(S_model, (3, 480 ,640))


train_ious,test_ious = learning_function(model,l_train,test)

plot(train_ious,test_ious)