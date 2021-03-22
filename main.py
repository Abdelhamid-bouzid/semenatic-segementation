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

images_train, labels_train, images_test, labels_test = load_data()

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(2)
model = model.float().to(device=device, dtype=torch.float)
#summary(model, (3, 480 ,480))

Loss_train,Loss_test = learning_function(model,images_train, labels_train,images_test,labels_test)

plot(Loss_train,Loss_test)