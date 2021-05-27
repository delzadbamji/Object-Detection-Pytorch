import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2 as cv

class training_dataset(Dataset):
    def __init__(self):
        '''
        read the training data and ground truth.
        implement your code here
        '''
        traindata=input("enter training data")
        # pdData = pd.read_csv('Dataset/trainingData.csv',sep=",",header=None)
        pdData = pd.read_csv(traindata, sep=",", header=None)
        trainground = input("enter training ground truth")
        # pdGround = pd.read_csv('Dataset/ground-truth.csv',sep=",",header=None)
        pdGround = pd.read_csv(trainground, sep=",", header=None)
        # print(self.featurestrain)
        # print("TRAIN   ", self.groundTruthtrain)
        #
        # self.featurestrain = self.featurestrain.to_numpy()
        # self.groundTruthtrain = self.groundTruthtrain.to_numpy()
        # self.featurestrain= torch.from_numpy(self.featurestrain)
        # self.groundTruthtrain= torch.from_numpy(self.groundTruthtrain)

        # self.featurestrain = np.asarray(self.featurestrain,dtype=np.float64)
        pdData = np.asarray(pdData,np.float64)
        self.len = len(pdData)
        # self.groundTruthtrain = np.asarray(self.groundTruthtrain,dtype=np.float64)
        pdGround = np.asarray(pdGround,dtype=np.float64)
        self.featurestrain = torch.Tensor(pdData)
        self.groundTruthtrain = torch.Tensor(pdGround)


    def __getitem__(self, item):
        return self.featurestrain[item], self.groundTruthtrain[item]

    def __len__(self):
        return self.len

