# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 20:41:38 2021

@author: Admin
"""
import torch
from torch.autograd import Function,Variable
from config import config

class loss_function(Function):
    def __init__(self):
        self.loss   = torch.nn.CrossEntropyLoss(reduce=False)
        self.SMOOTH = 1e-6
    
    def forward(self, pred, truth):
        loss = self.loss(pred, truth)
        
# =============================================================================
#         if config["number_classes"]==3:
#             device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             W = torch.zeros(truth.shape).to(device)
#             W = torch.where(truth==2,torch.tensor([0.5]).to(device), torch.tensor([0.25]).to(device))
#             loss = torch.mul(loss, W)
# =============================================================================
            
        loss = loss.mean()
        return loss