# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 17:23:34 2021

@author: Admin
"""
import torch 
import torchvision

class transformation():
    def __init__(self,size):
        self.size = size
        self.transforms = [torchvision.transforms.CenterCrop(self.size),
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomAffine(30),
                            torchvision.transforms.RandomCrop(self.size),
                            torchvision.transforms.RandomResizedCrop(self.size),
                            torchvision.transforms.RandomRotation(30),
                            ]
        self.FiveCrop   = torchvision.transforms.FiveCrop(self.size)
        self.resize     = torchvision.transforms.Resize(self.size)
        
    def apply_trans(self,all_tog):
        all_data_trans = []
        for trans in self.transforms:
            n_data = self.resize(trans(all_tog))
            all_data_trans.append(n_data)
        all_data_trans = torch.cat(all_data_trans)
        FiveCrop_data  = torch.cat(list(self.FiveCrop(all_tog)))
        all_data_trans = torch.cat([all_data_trans,FiveCrop_data])
        
        n_data   = all_data_trans[:,:3,:,:]
        n_labels = all_data_trans[:,3:,:,:]
        n_labels[n_labels<0.5]  = 0
        n_labels[n_labels>=0.5] = 1
        return n_data, n_labels