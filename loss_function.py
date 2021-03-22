# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:41:38 2021

@author: Admin
"""
import torch
from torch.autograd import Function

class loss_function(Function):
    def __init__(self):
        self.loss   = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred, truth,device):
        loss = self.loss(pred, truth.argmax(dim=1).to(device=device, dtype=torch.long))
        return loss