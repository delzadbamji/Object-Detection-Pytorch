import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from functions import overlapScore

import cv2
from cnn_model import *
from training_dataset import *
'''
In this part we will train the model.
'''
def train_model(net, dataloader, batchSize, lr_rate, momentum, Epoch_num):

    # implement your code here:
    criterion = nn.MSELoss()
    optimization = optim.SGD(net.parameters(),lr=lr_rate,momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimization, step_size=30, gamma=0.1)


    # implement your code here:

    for epoch in range(Epoch_num):
        scheduler.step()
        for i, data in enumerate(dataloader):
            optimization.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.view(batchSize, 1, 100, 100), labels.view(batchSize, 4)
            outputs = net(inputs)
            print(outputs," op")
            loss = criterion(outputs,labels)
            loss.backward()
            # scheduler.step()
            optimization.step()
            pbox = outputs.detach().numpy()
            gbox = labels.detach().numpy()
            print(pbox," pbox")
            print(gbox, " gbox")
            averageScore, scores = overlapScore(pbox,gbox)
            averageScore=averageScore/batchSize
            print("Score  ",averageScore/batchSize)
            print("loss  ", loss.item())
            print("epoch        ", epoch + 1, ", loss:  ", loss.item(), ", Average Score = ", averageScore)

        # print("epoch        ", epoch + 1, ", loss:  ", loss.item(), ", Average Score = ", averageScore)


if __name__ == '__main__':
    # implement your code here
    learning_rate = 0.0001
    momentum = 0.9
    batch = 50
    no_of_workers = 1
    shuffle = True
    epoch = 20
    traindataset = training_dataset()
    dataLoader = DataLoader(dataset=traindataset, batch_size=batch, shuffle=shuffle, num_workers=no_of_workers)
    model = cnn_model()
    model.train()
    train_model(model, dataLoader, batch,learning_rate, momentum, epoch)
    torch.save(model.state_dict(), 'model.pth')


