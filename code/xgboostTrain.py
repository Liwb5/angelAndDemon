from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.externals import joblib #joblib模块

import xgboost as xgb

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import sys
import os




def xgboost_cv(X, y, params, Kfold):
    """
    brief: 对xgboost模型进行交叉验证，并且画出ROC曲线。
    
    params:
        X: numpy类型。训练集
        y: numpy类型。每个训练集样本对应的标签
        params: dict类型。xgboost的参数
        Kfold: int类型。K折交叉验证。
        
    return: 
        xgb_models: K个交叉验证的模型。
    """
    random_state = np.random.RandomState(0)
    skf = StratifiedKFold(n_splits=Kfold,random_state=random_state) #k fold交叉验证
    i=0
    xgb_models = []
    for train_index, test_index in skf.split(X,y):
        xgb_model = xgb.XGBClassifier(booster=params['booster'],
                                      silent=params['silent'],
                                      #n_jobs=5, #不设置的话，自动获得最大线程数
                                      #以上为general params

                                      learning_rate = params['learning_rate'],
                                      min_child_weight = params['min_child_weight'], 

                                      max_depth=params['max_depth'], #
                                      max_delta_step = params['max_delta_step'], 
                                      gamma = params['gamma'],         
                                                         
                                      subsample = params['subsample'],    
                                      colsample_bytree=params['colsample_bytree'], #
                                     
                                      scale_pos_weight=params['scale_pos_weight'],
                                      #以上是booster的参数
                                      
                                      random_state = params['random_state'],
                                      n_estimators = params['n_estimators']
                                      )#树的棵树
        
        
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
        xgb_models.append(xgb_model)
    

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    plt.show()
    return xgb_models


def testModel(models, testX, versionSaved='100'):
    """
    brief: 将xgboost交叉验证的几个模型去预测测试集，并且对其取平均
    
    params:
        models: list类型。交叉验证的模型的集合。
        versionSaved: 保存模型预测结果的文件名。
    """
    i=0
    probas = []
    for xgb_final in models:
        predict = xgb_final.predict(testX)
        #查看有多少样本被预测为1
        print('samples have been predicted as positive samples: ', \
              sum(predict),'in %d model'%(i))
        
        probas_ = xgb_final.predict_proba(testX)
        probas.append(probas_[:,1])
        i+=1

    print(len(probas), len(probas[0]))
    predict = np.sum(probas, axis=0)/len(probas)
    #保存测试集的预测结果
    result = pd.read_csv('../originalDataset/exampleSubmission.csv')
    result.label = predict
    result.to_csv('../outputs/submission%s.csv'%(versionSaved), index=False)




def xgboostTrain(X, y, testX, versionSaved='100', params=None):
    """
    brief: 训练模型，并且用测试集测试，得到预测结果，保存预测结果,保存模型
    """
    xgb_final = xgb.XGBClassifier(booster=params['booster'],
                                  silent=params['silent'],
                                  #n_jobs=5, #不设置的话，自动获得最大线程数
                                  #以上为general params

                                  learning_rate = params['learning_rate'],
                                  min_child_weight = params['min_child_weight'], 

                                  max_depth=params['max_depth'], #
                                  max_delta_step = params['max_delta_step'], 
                                  gamma = params['gamma'],         

                                  subsample = params['subsample'],    
                                  colsample_bytree=params['colsample_bytree'], #

                                  scale_pos_weight=params['scale_pos_weight'],
                                  #以上是booster的参数

                                  random_state = params['random_state'],
                                  n_estimators = params['n_estimators']
                                 )#树的棵树
    xgb_final = xgb_final.fit(X, y, 
                     eval_set=[(X, y)],
                     eval_metric = "auc",
                     verbose = False)
    
    print(xgb_final.evals_result())
    predict = xgb_final.predict(testX)
    #查看有多少样本被预测为1
    print(sum(predict),'samples have been predicted as positive samples')
    probas_ = xgb_final.predict_proba(testX)
    probas = probas_[:,1]
    
    """
    #保存测试集的预测结果
    result = pd.read_csv('../originalDataset/exampleSubmission.csv')
    result.label = probas
    result.to_csv('../outputs/submission%s.csv'%(versionSaved), index=False)
    """
    print('saving result and model ...')
    #保存测试集的预测结果
    saveResult(probas)
    print('successful saving result!')
    """
    df1 = pd.DataFrame(np.arange(1, 1+len(probas)), index=None, columns=['Id'])
    df2 = pd.DataFrame(probas, index=None, columns=['label'])
    df = pd.concat([df1,df2],axis=1, ignore_index=True)
    df.to_csv('./outputs/submission.csv', index=False, )
    """
    #保存模型
    joblib.dump(xgb_final, './models/xgboost.model')
    print('successful saving model!')
    #加载模型
    #clf2 = joblib.load('./models/xgbModel_875dim.model')

def saveResult(y):
    index = np.array([i+1 for i in range(len(y))])
    index.resize(len(index), 1)
    y.resize(len(index), 1)
    data = np.concatenate((index, y), -1)

    df = pd.DataFrame(data, columns=['Id', 'label'])
    df[['Id']] = df[['Id']].astype(int)

    df.to_csv('./outputs/submission.csv', index=False)
    
    
if __name__ == "__main__":
    """
    filePath = os.path.dirname(__file__)
    if filePath != '':
        os.chdir(filePath) 
    """
    
    #所有参数的设置
    params = {'booster':'gbtree',
              'n_jobs': 5, #不设置的话，自动获得最大线程数
              #以上为general params
              
              'silent': False,
              'learning_rate': 0.1, #在xgboost的package中等价于eta参数
              'min_child_weight': 2, #控制过拟合，越大越不会过拟合
              'max_depth': 100,     #控制过拟合，越小越不会过拟合
              'max_delta_step': 1,  #数据不均衡的时候可以用
              'gamma': 0,   #模型在默认情况下，对于一个节点的划分
                            #只有在其loss function 得到结果大于0的情况下才进行，
                            #而gamma 给定了所需的最低loss function的值.
                            #所以gamma越大越保守（conservation）
              
              'subsample': 0.8,  #样本的选取比例，太大会过拟合，太小会欠拟合
              'colsample_bytree': 0.8, #特征的选取比例
              'scale_pos_weight': 340, #正负例的比例
              'random_state': 0,
              'n_estimators': 200  #树的数量
             }
    
    if len(sys.argv) >= 3: 
        #version = sys.argv[1]
        trainPath = sys.argv[1]
        testPath = sys.argv[2]
    else:
        trainPath = './dataAfterProcess/trainResult.csv'
        testPath = './dataAfterProcess/testResult.csv'
    
    outputPath = os.path.exists('./outputs/')
    modelPath = os.path.exists('./models/')
    if not outputPath:
        print('can not find the directory of ./outputs/')
        exit()
    if not modelPath:
        print('can not find the directory of ./models/')
        exit()
        
    #version = input("Please input the version of the dataset you want to load: ")
    
    #trainPath = '../dataAfterProcess/trainRes%s.csv'%(version)
    trainData = pd.read_csv(trainPath, header=None)
    X = trainData.values[:,0:trainData.shape[1]-1]
    y = trainData.values[:,trainData.shape[1]-1]

    #testPath = '../dataAfterProcess/testRes%s.csv'%(version)
    testData = pd.read_csv(testPath, header=None)
    testX = testData.values
    print(X.shape, y.shape, testX.shape)
    
    print('training model(it might take a few time)...')
    xgboostTrain(X, y, testX, params=params)
    
    """
    is_cv = input('if you want to do cross validation, please press y: ')
    
    if is_cv == 'y':
        xgb_models = xgboost_cv(X, y, params, 5)
        save_cv = input('if you want to save test result by using cv models, please press y: ')
        
        if save_cv == 'y':
            versionSaved = input('please input the file name(versionSaved) you want to save: ')
            testModel(xgb_models, testX, versionSaved)
    
    
    save_result = input('if you want to train the net and save the result, please press y: ')
    if save_result == 'y':
        versionSaved = input('please input the file name(versionSaved) you want to save: ')
        xgboostTrain(X, y, testX, versionSaved, params)
    """    
        
        
    
    
