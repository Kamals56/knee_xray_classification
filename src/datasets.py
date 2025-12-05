from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import pandas as pd
import torch


def read_xray(path):
    xray = cv2.imread(path.split(':')[0], cv2.IMREAD_GRAYSCALE)

    xray = xray.astype(np.float32) / 255

    xray_3ch = np.zeros((3, xray.shape[0], xray.shape[1]), dtype = xray.dtype) # (3, H, W)
    xray_3ch[0] = xray
    xray_3ch[1] = xray
    xray_3ch[2] = xray

    return xray_3ch
class Knee_Xray_Dataset(Dataset): 

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img = read_xray(self.dataset["Path"].iloc[index])
        label = self.dataset["KL"].iloc[index]

        img = torch.tensor(img, dtype=torch.float32)           # (3,H,W)
        label = torch.tensor(label, dtype=torch.long)

        res = {
            "img":img,
            "label": label
            }
        
        return res



