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
            proba = net.softmax(net(valid))
            predict = torch.Tensor.numpy(proba.data)
            result = predict.argmax(1) #将概率转成标签。
            score = roc_auc_score(validLabels,predict[:,1])
            print('epoch: %d, loss: %5f, auc: %5f predict: %d' % (epoch, loss.data[0],score, sum(result)))
       
    
    proba = net.softmax(net(valid)).data[:,1]
    predict = torch.Tensor.numpy(proba)
    fpr, tpr, thresholds = roc_curve(validLabels, predict)
    roc_auc = auc(fpr, tpr)
    #画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数能计算出来  
    plt.plot(fpr, tpr, lw=1)  
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')  
    plt.show()
    #torch.save(net.state_dict(), '../model/' + "score-{:.4f}-model.pkl".format(score))
    
    
    
#神经网络训练，并且用测试集测试，保存测试集的结果
def NetTrain(x_data, y_data, x_valid, y_valid,
            testData, version, versionSaved='100'):
    
    features = x_data.shape[1]
    net = Net(features)
    
    weight = [0.003,0.997]
    maxNumEpoch = 100
    learning_rate = 0.001
    
    #-------training -------------#
    train(net,x_data,y_data,x_valid,y_valid,\
          lr=learning_rate,weight=weight,maxNumEpoch=maxNumEpoch)
    
    x_test = Variable(torch.from_numpy(testData).float(), requires_grad=True)
    prob = net.softmax(net(x_test)).data[:,1].numpy().reshape((-1,1))
    
    result = pd.read_csv('./originalDataset/exampleSubmission.csv')
    result.label = prob
    result.to_csv('./outputs/submission%s.csv'%(versionSaved),index=False) 
    

    
#对xgboost模型进行交叉验证，并且画出ROC曲线。
def xgboost_cv(X, y, xgboost, Kfold):
    random_state = np.random.RandomState(0)
    skf = StratifiedKFold(n_splits=Kfold,random_state=random_state) #k fold交叉验证
    i=0
    
    for train_index, test_index in skf.split(X,y):
        xgb_model = xgboost
        xgb_model = xgb_model.fit(X[train_index], y[train_index], 
                                  eval_set=[(X[train_index], y[train_index]),
                                            (X[test_index], y[test_index])],
                                  eval_metric = "auc",
                                  verbose = False)

        #evals_result = xgb_model.evals_result()
        #print(evals_result)
        probas_ = xgb_model.predict_proba(X[test_index])

        #[:,1]二分类有0的概率，也有预测为1的概率，这里提取预测为1的概率
        fpr, tpr, thresholds = roc_curve(y[test_index], probas_[:,1])
        roc_auc = auc(fpr, tpr)
        score = roc_auc_score(y[test_index] , probas_[:,1])#验证集的auc分数

        train_probas = xgb_model.predict_proba(X[train_index])
        train_score = roc_auc_score(y[train_index], train_probas[:,1])#训练集的auc分数
        print("auc_test: %5f,auc_train:%5f in %d fold. index shape:%d"\
              %(score, train_score, i, len(train_index))) 

        plt.plot(fpr, tpr, lw=1, alpha=0.8,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1
    
    

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    plt.show()

    
    
#训练模型，并且用测试集测试，得到预测结果，保存预测结果
def xgboostTrain(X, y, testX, versionSaved='100', params=None):
    
    xgb_final = xgb.XGBClassifier(scale_pos_weight=340,
                        n_jobs=5,
                        max_depth=5,
                        min_child_weight=1,
                        learning_rate = 0.1,
                        #random_state = random_state,
                        gamma = 0,
                        subsample = 0.8,
                        colsample_bytree = 0.7,
                        n_estimators = 50)  
    
    xgb_final = xgb_final.fit(X, y, 
                     eval_set=[(X, y)],
                     eval_metric = "auc",
                     verbose = False)
    
    print(xgb_final.evals_result())
    predict = xgb_final.predict(testX)
    #查看有多少样本被预测为1
    print(sum(predict),' samples have been predicted as positive samples')
    probas_ = xgb_final.predict_proba(testX)
    probas = probas_[:,1]
    
    #保存测试集的预测结果
    result = pd.read_csv('./originalDataset/exampleSubmission.csv')
    result.label = probas
    result.to_csv('./outputs/submission%s.csv'%(versionSaved), index=False)
    
    
if __name__ == "__main__":
    os.chdir('/home/liwb/Documents/projects/angelAndDemon/')
    version = sys.argv[1]
    trainPath = './dataAfterProcess/trainRes%s.csv'%(version)
    trainData = pd.read_csv(trainPath, header=None)
    print('trainData shape: ',trainData.shape)
    #X = trainData.values[:,0:trainData.shape[1]-1]
    #y = trainData.values[:,trainData.shape[1]-1]
    
        
    #x_data = trainData.values[:,0:trainData.shape[1]-1]
    #y_data = trainData.values[:,trainData.shape[1]-1]
    #x_valid = validData.values[:,0:validData.shape[1]-1]
    #y_valid = validData.values[:,validData.shape[1]-1]
    #print(x_data.shape, x_valid.shape)
    #print(sum(y_data), sum(y_valid))
    #print(y_data.shape, y_valid.shape)
    
    input("press any key to continue: ")
    
    testPath = "./dataAfterProcess/testRes%s.csv"%(version)
    testData = pd.read_csv(testPath, header=None)
    
    
    
    
    
    

    
