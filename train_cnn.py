import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from functions import overlapScore


from cnn_model import *
from training_dataset import *
'''
In this part we will train the model.
'''
def train_model(net, dataloader, batchSize, lr_rate, momentum, Epoch_num):
    '''
    training process of this model
    :param net: model
    :param dataloader: dataloader
    :param batchSize: batch size
    :param lr_rate: learning rate
    :param momentum: momentum
    :param Epoch_num: epoch number
    :return:
    '''
    '''
    setup loss function(mean squared error loss)
    optimization (stochastic gradient descent)
    scheduler(hint: optim.lr_scheduler.StepLR(), step_size=30, gamma=0.1)
    '''
    # implement your code here:
    criterion = nn.MSELoss()
    optimization = optim.SGD(net.parameters(),lr=lr_rate,momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimization, step_size=30, gamma=0.1)


    '''
    loop for training
    '''
    # implement your code here:

    for epoch in range(Epoch_num):

        scheduler.step()
        score=0
        for i, data in enumerate(dataloader):
            # clear the gradients for all optimized variables
            optimization.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            inputs, labels = data
            inputs, labels = inputs.view(batchSize, 1, 100, 100), labels.view(batchSize, 4)
            outputs = net(inputs)
            # print(outputs," op")
            # calculate the loss
            loss = criterion(outputs,labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            # scheduler.step()
            optimization.step()
            # calculate the score using overlapScore function
            pbox = outputs.detach().numpy()
            gbox = labels.detach().numpy()
            # print(pbox," pbox")
            # print(gbox, " gbox")
            averageScore, scores = overlapScore(pbox,gbox)
            # print(scores)
            averageScore=averageScore/batchSize
            score = averageScore if averageScore>score else score
            # print("Score  ",averageScore/batchSize)
            # print("loss  ", loss.item())
        print("epoch        ", epoch + 1, ", loss:  ", loss.item(), ", Average Score = ", score)

        # print("epoch        ", epoch + 1, ", loss:  ", loss.item(), ", Average Score = ", averageScore)

        '''
        print out epoch, loss and average score in following format
        epoch     1, loss: 426.835693, Average Score = 0.046756
        '''


    print('Finish Training')


if __name__ == '__main__':
    # hyper parameters
    # implement your code here
    learning_rate = 0.000001 #0.000001
    momentum = 0.9
    batch = 100
    no_of_workers = 2
    shuffle = True
    epoch = 50
    print("please ignore the warning after entering the paths and wait.......")
    # load dataset
    # implement your code here
    traindataset = training_dataset()
    # setup dataloader
    # implement your code here
    # dataLoader = DataLoader(traindataset, batch, shuffle, no_of_workers)
    dataLoader = DataLoader(dataset=traindataset, batch_size=batch, shuffle=shuffle, num_workers=no_of_workers)
    model = cnn_model()
    model.train()

    train_model(model, dataLoader, batch,learning_rate, momentum, epoch)
    # save model
    torch.save(model.state_dict(), 'model.pth')


