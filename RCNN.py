# -*- coding: utf-8 -*-
"""
@author: noahb
"""

#training and testing a model based on the VGG16 CNN

#import necessary Libraries
import torch
from torch import nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torchvision.datasets import ImageFolder
from torchvision.transforms import Grayscale 
from torchvision.transforms import Compose

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
    ToTensor()
    ])
    
dataset_train = ImageFolder(root="./archive/train", transform=transform)
dataloader_train = DataLoader(dataset_train, batch_size=256,shuffle=True)
#print(next(iter(dataloader_train)).shape)


dataset_test = ImageFolder(root="./archive/test", transform=transform)
dataloader_test = DataLoader(dataset_test, batch_size=256, shuffle=True)
#print(next(iter(dataloader_test)).shape)
        
        
        

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, padding = 1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=30, out_channels=30, kernel_size=7, padding = 2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(in_channels=30, out_channels=30, kernel_size=11, padding = 3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Dropout(.5),
    nn.Flatten(),
    nn.Linear(in_features=270, out_features=256),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(in_features=256, out_features=128),
    nn.ReLU(),
    nn.Dropout(.5),
    nn.Linear(in_features=128, out_features=7)
    )

print(model)
#print(models.AlexNet())
model.load_state_dict(torch.load("./Finalmodel40"))
trainNetwork(net=model, num_epochs=20, learning_rate=.1, trainload=dataloader_train)
testNetwork(net=model, testload=dataloader_test)
torch.save(model.state_dict(), "./Finalmodel60")


