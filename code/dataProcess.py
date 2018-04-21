import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from sklearn import metrics
from sklearn.metrics import roc_auc_score as auc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
import sys
import os

if __name__ == "__main__":
    #----------parameters---------------#
    os.chdir('/home/liwb/Documents/projects/angelAndDemon/')
    
    trainPath = "./originalDataset/train.csv"
    testPath = "./originalDataset/test.csv"
    Thresh = 10000           #每一列非nan值超过Thresh才会保留
    Frequency = 100000        #每一列，某元素出现次数超过Frequency的，那一列会被丢弃
    version = '5'            #数据处理后保存成文件的文件版本号
    version = sys.argv[1]

    #读取原始数据
    trainData = pd.read_csv(trainPath, header=None)
    testData = pd.read_csv(testPath, header=None)

    print('trainData shape: ', trainData.shape)
    print('testData shape: ', testData.shape)

    featuresNum = trainData.shape[1]#得到列数
    trainNum = trainData.shape[0]#得到行数

    #为了添加列名，只好先转成numpy在转回dataframe
    trainData = trainData.values
    testData = testData.values

    label = trainData[:,featuresNum-2].reshape((-1,1))#提取出标签
    label = np.where(label==-1,0,1).reshape((-1,1)) #将标签转从{-1,1}换成{0,1}
    print("label shape: ", label.shape, "sum label: ",sum(label))


    #合并训练集和测试集，方便后面进行数据处理
    data = np.vstack((trainData[:,0:(featuresNum-2)],testData[:,0:(featuresNum-2)]))
    data = pd.DataFrame(data)
    print('merge data shape: ',data.shape)


    print("removing innumeric features: ")
    #将非数值型数据转换成nan值
    #coerce参数将非数字的值转换成nan，如果是ignore参数，则不处理       
    data1 = data.apply(lambda x:pd.to_numeric(x, errors="coerce")) 

    data1 = data1.dropna(axis=1, thresh=Thresh)#删除nan很多的列
    print('trainData shape: ', data1.shape)


    #缺失值填充
    data2 = data1.fillna(data1.mean()) #对nan值用该列的均值补全。


    #对于那些一列中某个元素出现次数太多的，也将这些列删除
    index = []
    data3 = data2.values
    data3 = pd.DataFrame(data3)
    for i in range(data3.shape[1]):
        freq = data3[i].value_counts().values[0]
        if freq>Frequency:
            index.append(i)#获得重复元素超过50000的下标
    print("indexes that will be removed:", index)
    data4 = data3.values
    data4 = np.delete(data4, index, axis=1)#删除index下标的所有列

    print("data4 shape: ",data4.shape)


    #归一化处理
    mmScaler = pp.MinMaxScaler()
    data4 = mmScaler.fit_transform(data4)
    print("regulization , data4 shape:", data4.shape)



    testRes = data4[trainNum:]
    trainRes = data4[0:trainNum]
    print("testRes shape:",testRes.shape)
    print("trainRes shape:",trainRes.shape)
    #保存测试集
    testRes = pd.DataFrame(testRes)
    testRes.to_csv('./dataAfterProcess/testRes%s.csv'%(version), header=False, index=False, encoding='utf-8')

    #分割验证集
    trainRes1 = np.hstack((trainRes, label))
    print("trainRes shape after merge label: ",trainRes1.shape)
    np.random.shuffle(trainRes1)


    validRes = trainRes1[0:20000]
    trainRes2 = trainRes1[20000:]
    print("validRes shape: ",validRes.shape, "trainRes2 shape: ",trainRes2.shape)


    #保存训练集
    trainRes2 = pd.DataFrame(trainRes2)
    trainRes2.to_csv('./dataAfterProcess/trainRes%s.csv'%(version), header=False, index=False, encoding='utf-8')

    #保存验证集
    validRes = pd.DataFrame(validRes)
    validRes.to_csv('./dataAfterProcess/validRes%s.csv'%(version), header=False, index=False, encoding='utf-8')









