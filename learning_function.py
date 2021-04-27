# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 21:23:32 2021

@author: Admin
"""

from loss_function import loss_function
import torch
import torch.nn.functional as F
from config import config
from IOU import iou_pytorch,iou_numpy
import numpy as np
from torch.utils.data import DataLoader
from RandomSampler import RandomSampler
import math

def learning_function(model,l_train,test):
    
    
    ''' #################################################  set up optim  ################################################### '''
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = model.to(device)
    model.train()
    
    optimizer     = torch.optim.Adam(model.parameters(),lr = config['learning_rate'])
    
    ''' #################################################  Dtat loaders  ################################################### '''
    train_sampler=RandomSampler(len(l_train), config["iteration"] * config["batch_size"])
    l_loader = DataLoader(l_train, config["batch_size"],drop_last=True,sampler=train_sampler)
    
    test_loader = DataLoader(test, config["batch_size"],drop_last=False)
    
    ''' #################################################  initialization  ################################################### '''
        
    Loss_train,Loss_test,train_ious,test_ious = [],[],[],[]
    best_iou  = 0
    iteration = 0
    train_iou = []
    for l_input, l_target in l_loader:
        
        iteration += 1
        
        l_input, l_target = l_input.to(device).float(), l_target.to(device).long()
        
        
        outputs = model(l_input)
        
        loss = loss_function().forward(outputs, l_target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        iou = iou_pytorch(outputs.argmax(dim=1).cpu(),l_target.argmax(dim=1).cpu())
        train_iou.append(iou)
        
        if iteration%config["decay_lr_iter"]==0:
            if optimizer.param_groups[0]["lr"]>config["min_lr"]:
                optimizer.param_groups[0]["lr"] *= config["decay_lr"]
        
        print('##########################################################################################################')
        print("   #####  Train iteration: {} train_loss: {:0.4f} train_iou: {:0.4f}".format(iteration,loss.item(),iou))
        print('##########################################################################################################')
        
        if iteration%config["test_model_cycel"]==0:
            model.eval()
            test_iou  = 0
            with torch.no_grad(): 
                for l_input, l_target in test_loader:
                    l_input, l_target = l_input.to(device).float(), l_target.to(device).long()
                    
                    outputs = model(l_input)
                    
                    iou = iou_pytorch(outputs.argmax(dim=1).cpu(),l_target.argmax(dim=1).cpu())
                    test_iou += iou
                    
            test_iou = test_iou/len(test)
            test_ious.append(test_iou)
            train_ious.append(sum(train_iou)/len(train_iou))
            train_iou = []
            print('**********************************************************************************************************')
            print("   #####  Train iteration: {} test_iou: {:0.4f} ".format(iteration,iou))
            print('**********************************************************************************************************')
        
            if test_iou>best_iou:
                best_iou = test_iou
                torch.save(model,'models/model.pth')
            
        
        model.train()
    return train_ious,test_ious