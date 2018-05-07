import pandas as pd
import numpy as np
from sklearn import preprocessing as pp
from sklearn import metrics
from sklearn.metrics import roc_auc_score as auc
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import sys
import os

def read_data(trainPath, testPath):
    #读取原始数据
    trainData = pd.read_csv(trainPath, header=None)
    testData = pd.read_csv(testPath, header=None)

    print('trainData shape: ', trainData.shape)
    print('testData shape: ', testData.shape)

    return trainData, testData


def process_nonNumeric(data):
    """
    brief：处理数据中包含的非数值类型，将其映射成one-hot形式。
        
    params：
        data: DataFrame类型。包含非数值型的列，需要被处理。
    
    return：
        data: DataFrame类型，删除了原始data中非数值的列
        newData：DataFrame类型。将非数值的列映射成one-hot的形式后产生的DataFrame类型。
    """
    nonNumericCol = [132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 
                     162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 
                     192, 195, 198, 201, 207, 244] 
    
    nonNumericData = data[nonNumericCol]#提取非数值的列

    count = 0
    newData = pd.DataFrame()
    for col in nonNumericData.columns:
        #将非数值的每一列修改成one-hot的形式
        res = nonNumericData[col].value_counts()
        for indx in res.index:
            newData[count] = nonNumericData[col].apply(lambda x, index: 1 if x==index else 0,
                                                      args=(indx,))
            count += 1
    
    res = np.where(nonNumericData.isnull(), 1, 0)
    
    newData = np.hstack((newData.values, res))
    newData = pd.DataFrame(newData)
    
    #将非数值的列删除
    data = data.drop(nonNumericCol,axis=1)

    print('nonNumericData shape: ', newData.shape)
    
    return data, newData


def process_date(data):
    """
    brief:
        处理数据中的日期，更改时间日期格式
    
    params:
        data: DataFrame类型。包含日期的列，需要被处理。
        
    return: 
        data: DataFrame类型，删除了原始data中日期所在的列
        date: DataFrame类型。将日期所在的列按照年月日时分秒映射成6个独立的属性。
    """
    dateCol = 206  #时间所在的列，手动统计的
    
    date = pd.DataFrame()
    date['date'] = pd.to_datetime(pd.Series(data[dateCol]), format='%Y-%m-%d-%H.%M.%S.%f')
    
    # 转化为6个单独的属性
    date['Year'] = date['date'].apply(lambda x: x.year)
    date['Month'] = date['date'].apply(lambda x: x.month)
    date['Day'] = date['date'].apply(lambda x: x.day)
    date['Hour'] = date['date'].apply(lambda x: x.hour)
    date['Minute'] = date['date'].apply(lambda x: x.minute)
    date['Second'] = date['date'].apply(lambda x: x.second)
    
    data = data.drop(dateCol, axis=1)
    date = date.drop('date', axis=1)
    
    return data, date
    

def remove_NAN(data, Thresh):
    """
    brief: 将data的每一列中非nan值少于Thresh的列删除。
    
    params:
        data: DataFrame类型。
        Thresh: int, 非nan值少于Thresh的列将被删除
    """

    data = data.dropna(axis=1, thresh=Thresh)#删除nan很多的列
    print('data shape after remove NAN columns: ', data.shape)
    
    return data

def remove_cols_appear_too_much(data, frequency):
    #对于那些一列中某个元素出现次数太多的，也将这些列删除
    index = []
    data3 = data.values
    data3 = pd.DataFrame(data3)
    for i in range(data3.shape[1]):
        freq = data3[i].value_counts().values[0]
        if freq>frequency:
            index.append(i)#获得重复元素超过frequency的下标
    print("indexes that will be removed:", index)
    data4 = data3.values
    data4 = np.delete(data4, index, axis=1)#删除index下标的所有列

    print("data shape after remove_cols_appear_too_much: ",data4.shape)
    return data4

def remove_nonNumeric(data, Thresh=10000):
    """
    brief: 将非数值型数据转换成nan值,并且删除非nan值不超过Thresh的列。
    params: 
        data: DataFrame类型。
        Thresh: int, 非nan值少于Thresh的列将被删除
    """
    #将非数值型数据转换成nan值
    #coerce参数将非数字的值转换成nan，如果是ignore参数，则不处理       
    data = data.apply(lambda x:pd.to_numeric(x, errors="coerce")) 

    data = data.dropna(axis=1, thresh=Thresh)#删除nan很多的列
    print('trainData shape: ', data.shape)
    
    return data


def norm(data):
    #归一化处理
    mmScaler = pp.MinMaxScaler()
    data4 = mmScaler.fit_transform(data)
    print("data shape after normalization:", data4.shape)
    return data4


#----------parameters---------------#


Threshold = 10000           #每一列非nan值超过Thresh才会保留
#Frequency = 100000        #每一列，某元素出现次数超过Frequency的，那一列会被丢弃
version = '50'       


if __name__ == "__main__":
    """
    usage:
        cd /path/to/code/  #进入代码文件夹
        mkdir dataAfterProcess/  #新建文件夹，用于存放处理好的数据
        python dataProcess.py  /path/to/train.csv  /path/to/test.csv #执行数据处理代码，两个参数分别是训练集与测试集的所在的路径

    """
    """ 
    path = os.path.dirname(__file__) #获得本文件所在的目录
    if path != "":
        os.chdir(path) #将当前路径设置为本文件所在的目录，方便下面读取文件。
    """
    
    if len(sys.argv) >= 3:
        #version = sys.argv[1]
        trainPath = sys.argv[1]
        testPath = sys.argv[2]
    

    #trainPath = "../originalDataset/train.csv"
    #testPath = "../originalDataset/test.csv"    
    
    trainData, testData = read_data(trainPath, testPath)

    featuresNum = trainData.shape[1]#得到列数
    trainNum = trainData.shape[0]#得到行数

    #为了添加列名，只好先转成numpy再转回dataframe
    trainData = trainData.values
    testData = testData.values

    label = trainData[:,featuresNum-2].reshape((-1,1))#提取出标签
    label = np.where(label==-1,0,1).reshape((-1,1)) #将标签转从{-1,1}换成{0,1}
    print("label shape: ", label.shape, "sum label: ",sum(label))

    #合并训练集和测试集，方便后面进行数据处理
    data = np.vstack((trainData[:,0:(featuresNum-2)],testData[:,0:(featuresNum-2)]))
    data = pd.DataFrame(data)
    print('merge data shape: ',data.shape)
    
    print('processing data with nonNumeric data (it might take a few minutes)...')
    data, nonNumericData = process_nonNumeric(data)

    data, dateData = process_date(data)
    
    #将其它列中的非数值数据转换成nan值。
    #coerce参数将非数字的值转换成nan，如果是ignore参数，则不处理
    #data = data.apply(lambda x:pd.to_numeric(x, errors="coerce"))
    
    #删除NAN值很多的列
    #data = remove_NAN(data, Threshold)

    print(data.shape)
    #缺失值填充
    #data = data.fillna(data.mean()) #对nan值用该列的均值补全。

    #删除某个元素出现次数太多的列
    #data = remove_cols_appear_too_much(data, Frequency)

    #归一化处理
    #data = norm(data)
    
    #合并所有的数据
    data = np.hstack((data.values, nonNumericData.values))
    print('merge data, nonNumericData and dateData: ',data.shape)

    testRes = data[trainNum:]
    trainRes = data[0:trainNum]
    print("testRes shape:",testRes.shape)
    print("trainRes shape:",trainRes.shape)
    
    print('saving data (it might take a few minutes)...')
    #保存测试集
    testRes = pd.DataFrame(testRes)
    testRes.to_csv('./dataAfterProcess/testResult.csv', header=False, index=False, encoding='utf-8')
    
    #保存训练集
    trainRes1 = np.hstack((trainRes, label))
    trainRes2 = pd.DataFrame(trainRes1)
    trainRes2.to_csv('./dataAfterProcess/trainResult.csv', header=False, index=False, encoding='utf-8')
    print('successful saving data!')

    


