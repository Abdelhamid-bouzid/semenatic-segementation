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
l_train = load_data("data", "l_train")
test    = load_data("data", "test")

#####################################################################################################
#################################### student model ##################################################
#####################################################################################################
model = UNet(2)
#summary(S_model, (3, 480 ,640))


Loss_train,Loss_test = learning_function(model,l_train,test)

plot(Loss_train,Loss_test)