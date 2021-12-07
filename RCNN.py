# -*- coding: utf-8 -*-
"""
@author: noahb
"""

#training and testing a model based on the VGG16 CNN

#import necessary Libraries
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.transforms import Grayscale 
from torchvision.transforms import Compose
import torchvision.models as models

def trainNetwork(net, num_epochs, learning_rate, trainload):
    start_time = time.time()
    optimizer = optim.SGD(net.parameters(),lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    
    for i in range(num_epochs):
        #correct, total = 0,0
        training_loss = 0.0
        for data, label in trainload:
            optimizer.zero_grad()
            output = net(data)
            '''
            for o,l in zip(torch.argmax(output,axis = 1),label):
                if o == l:
                    correct += 1
                total += 1
                '''
            loss = loss_func(output,label)
            loss.backward()
            optimizer.step()
            training_loss+= (loss.item() * data.size(0))
        print(f'Epoch: {i+1} / {num_epochs} \t\t\t Training Loss:{training_loss/len(trainload)}')
        #print(f'Correct Predictions: {correct}/{total}')
    elapsed = time.time() - start_time
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))
    return loss

def testNetwork(net,testload):
    test_loss = 0.0
    correct, total = 0,0
    loss_func = nn.CrossEntropyLoss()
    for data,label in testload:
 
        output = net(data)
        for o,l in zip(torch.argmax(output,axis = 1),label):
            if o == l:
                correct += 1
            total += 1
        loss = loss_func(output,label)
        test_loss += loss.item() * data.size(0)
    print(f'Testing Loss:{test_loss/len(testload)}')
    print(f'Correct Predictions: {correct}/{total}')

    
transform = Compose([
    Grayscale(),
    Resize(224),
    ToTensor()
    ])
    
dataset_train = ImageFolder(root="./archive/train", transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=256,shuffle=True)
#print(next(iter(dataloader_train)).shape)


dataset_test = ImageFolder(root="./archive/test", transform=transform)
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)
#print(next(iter(dataloader_test)).shape)
        
        
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
Alexnet = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(4096, 7)
    )

model = Alexnet
'''
model = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                    nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 7))
'''
print(model)
#print(models.AlexNet())

trainNetwork(net=model, num_epochs=40, learning_rate=.1, trainload=dataloader_train)
testNetwork(net=model, testload=dataloader_test)
torch.save(model.state_dict(), "./model")


