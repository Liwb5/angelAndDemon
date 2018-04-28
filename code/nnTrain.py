import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import sys
import os


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
    
    
     
def train(net,inputs,labels,fold, validData='unused',validLabels='unused',
          lr=0.001,weight=[0.5,0.5],maxNumEpoch=200):
    
    criterion = nn.CrossEntropyLoss(torch.Tensor(weight))
    optimizer = optim.Adam(net.parameters(), lr=lr)
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    inputs, Labels = Variable(torch.from_numpy(inputs).float(),requires_grad=True),\
                        Variable(torch.from_numpy(labels)).view(-1).long()
    
    if validData != 'unused':
        valid = Variable(torch.from_numpy(validData).float(),requires_grad=True)

    for epoch in range(maxNumEpoch):  # loop over the dataset multiple times 
        # zero the parameter gradients
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, Labels)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 and validData != 'unused':
            proba = net.softmax(net(valid))
            predict = torch.Tensor.numpy(proba.data)
            result = predict.argmax(1) #将概率转成标签。
            score = roc_auc_score(validLabels,predict[:,1])
            
            proba = net.softmax(net(inputs))
            proba = torch.Tensor.numpy(proba.data)
            train_score = roc_auc_score(labels,proba[:,1])
            print('fold: %d, epoch: %d, loss: %5f, valid_auc: %5f train_auc: %5f, predict: %d' % \
                  (fold, epoch, loss.data[0], score, train_score, sum(result)))
    return net
   

    
def nnTrain_cv(X, y, params, Kfold):
    """
    brief: 进行交叉验证训练
    
    return: 
        nets: 返回交叉验证所得到的Kfold个模型
    """
    random_state = np.random.RandomState(0)
    skf = StratifiedKFold(n_splits=Kfold,random_state=random_state) #k fold交叉验证
    
    i=0
    nets = []
    for train_index, valid_index in skf.split(X, y):
        net = Net(X.shape[1])
        net = train(net = net, 
                    inputs = X[train_index], 
                    labels = y[train_index], 
                    validData = X[valid_index], 
                    validLabels = y[valid_index],
                    lr = params['lr'], 
                    weight = params['weight'], 
                    maxNumEpoch = params['maxNumEpoch'],
                    fold = i)
        
        valid = Variable(torch.from_numpy(X[valid_index]).float(),requires_grad=True)
        proba = net.softmax(net(valid)).data[:,1]
        proba = torch.Tensor.numpy(proba)
        fpr, tpr, thresholds = roc_curve(y[valid_index], proba)
        plt.plot(fpr, tpr, lw=1, alpha=0.8) 
        
        nets.append(net)
        i += 1
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    plt.show()
        
    return nets   
        

def NetTrain(x_data, y_data, testData, params, versionSaved='100'):
    """
    brief: 对训练集进行训练，并且用测试集测试，保存测试集的结果
    """
    features = x_data.shape[1]
    net = Net(features)
    
    #-------training -------------#
    train(net,x_data,y_data,
          fold = 0,
          lr=params['lr'],
          weight=params['weight'],
          maxNumEpoch=params['maxNumEpoch'])
    
    x_test = Variable(torch.from_numpy(testData).float(), requires_grad=True)
    prob = net.softmax(net(x_test)).data[:,1].numpy().reshape((-1,1))
    
    result = pd.read_csv('../originalDataset/exampleSubmission.csv')
    result.label = prob
    result.to_csv('../outputs/submission%s.csv'%(versionSaved),index=False) 
    
    
if __name__ == '__main__':
    filePath = os.path.dirname(__file__)
    if filePath != '':
        os.chdir(filePath) 
        
    params = {'weight':[0.003,0.997],
              'maxNumEpoch': 10,
              'lr': 0.001 }
        
    version = input("Please input the version of the dataset you want to load: ")
    is_cv = input('if you want to do cross validation, please press y: ')
    
    trainPath = '../dataAfterProcess/trainRes%s.csv'%(version)    
    trainData = pd.read_csv(trainPath, header=None)
    X = trainData.values[:,0:trainData.shape[1]-1]
    y = trainData.values[:,trainData.shape[1]-1]

    testPath = '../dataAfterProcess/testRes%s.csv'%(version)
    testData = pd.read_csv(testPath, header=None)
    testX = testData.values
    print(X.shape, y.shape, testX.shape)
    
    
    if is_cv == 'y':
        nnTrain_cv(X, y, params, 5)
        
    save_result = input('if you want to train the net and save the result, please press y: ')
    
    if save_result == 'y':
        versionSaved = input('please input the file name(versionSaved) you want to save: ')
        NetTrain(X, y, testX, params, versionSaved = versionSaved)
    
    
    
    
    