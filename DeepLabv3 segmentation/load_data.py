# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 10:55:48 2021

@author: Admin
"""
import numpy as np
import os


class load_data:
    def __init__(self, path):
        self.dataset = np.load(path, allow_pickle=True)

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        return image, label

    def __len__(self):
        return len(self.dataset["images"])