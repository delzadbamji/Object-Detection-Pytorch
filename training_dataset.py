import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

class training_dataset(Dataset):
    def __init__(self):

        traindata=input("enter t data")
        dat = pd.read_csv(traindata)
        trainground = input("enter t gt")
        gnd = pd.read_csv(trainground)

        # print(self.featurestrain)
        # print("TRAIN   ", self.groundTruthtrain)


        # self.featurestrain = dat
        self.featurestrain = np.asarray(dat)
        self.featurestrain = torch.Tensor(self.featurestrain)
        self.len = len(np.asarray(dat))
        self.groundTruthtrain = torch.Tensor(np.asarray(gnd,dtype=np.float64))


    def __getitem__(self, item):
        return self.featurestrain[item], self.groundTruthtrain[item]

    def __len__(self):
        return self.len

