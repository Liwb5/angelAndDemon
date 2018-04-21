import torch
from torch.autograd import Variable
from torch.autograd import Function

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score as auc
from sklearn import preprocessing

import numpy as np
import pandas as pd

import sys


class Net(nn.Module):
    def __init__(self, features):
        super(Net,self).__init__()   
        self.fc1 = nn.Linear(features, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        #prob = self.softmax(x)
        return x #,prob.data[:,1].numpy()
     
def train(net,inputs,labels,validData=None,validLabels=None,lr=0.001,weight=[0.5,0.5],maxNumEpoch=200):
    criterion = nn.CrossEntropyLoss(torch.Tensor(weight))
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    inputs, labels = Variable(torch.from_numpy(inputs).float(),requires_grad=True),\
                        Variable(torch.from_numpy(labels)).view(-1).long()
    valid = Variable(torch.from_numpy(validData).float(),requires_grad=True)

    for epoch in range(maxNumEpoch):  # loop over the dataset multiple times 
        # zero the parameter gradients
        optimizer.zero_grad()
        
        outputs = net(inputs)
    #    print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            predict = torch.Tensor.numpy(net.softmax(net(valid)).data[:,1])
            score = auc(validLabels,predict)
            print('epoch: %d, loss: %5f, auc: %5f predict: %d' % (epoch, loss.data[0],score, sum(predict)))
        
    #torch.save(net.state_dict(), '../model/' + "score-{:.4f}-model.pkl".format(score))  
    
    
    
if __name__ == "__main__":
    version = sys.argv[1]
    trainPath = '../data/trainRes%s.csv'%(version)
    validPath = '../data/validRes%s.csv'%(version)
    
    
    
    trainData = pd.read_csv(trainPath)
    validData = pd.read_csv(validPath)
    x_data = trainData.values[:,0:trainData.shape[1]-1]
    y_data = trainData.values[:,trainData.shape[1]-1]
    x_valid = validData.values[:,0:validData.shape[1]-1]
    y_valid = validData.values[:,validData.shape[1]-1]
    #print(y_data)
    #a = input("press any key to continue: ")
    print(x_data.shape, x_valid.shape)

    print(sum(y_data), sum(y_valid))
    print(y_data.shape, y_valid.shape)
    
    features = x_data.shape[1]
    print("feature: ",features)
    net = Net(features)
    
    weight = [0.003,0.997]
    maxNumEpoch = 100
    learning_rate = 0.001
    targetPath = '../data/pytotrch_model2.csv'

    #-------training -------------#
    train(net,x_data,y_data,x_valid,y_valid,lr=learning_rate,weight=weight,maxNumEpoch=maxNumEpoch)
    
    

    #-------testing & saving ------------#
    testPath = '../data/testRes%s.csv'%(version)
    testData = pd.read_csv(testPath)
    testData = testData.values
    print(testData.shape)
    a=input()
    x_test = Variable(torch.from_numpy(testData[:]).float(),requires_grad=True)
    prob = net.softmax(net(x_test)).data[:,1].numpy().reshape((,1))
    c = pd.DataFrame(prob)
    c.to_csv(targetPath,index=False,encoding = 'utf-8')
    
    
    
    
    
    
    
    
    
    
    
    
