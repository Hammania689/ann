
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#读入文件
path1='C:\\Users\\w946225.GP0000231812\\Coureses_Haipeng_Tang\\CSC 611\\Project2\\dementia_prediciton\\MainProject\\project\\data\\train_relief.csv'
path2='C:\\Users\\w946225.GP0000231812\\Coureses_Haipeng_Tang\\CSC 611\\Project2\\dementia_prediciton\\MainProject\\project\\data\\test_relief.csv'
path3='C:\\Users\\w946225.GP0000231812\\Coureses_Haipeng_Tang\\CSC 611\\Project2\\dementia_prediciton\\MainProject\\project\\data\\train_infor.csv'
path4='C:\\Users\\w946225.GP0000231812\\Coureses_Haipeng_Tang\\CSC 611\\Project2\\dementia_prediciton\\MainProject\\project\\data\\test_infor.csv'
path5='C:\\Users\\w946225.GP0000231812\\Coureses_Haipeng_Tang\\CSC 611\\Project2\\dementia_prediciton\\MainProject\\project\\data\\train_rf.csv'
path6='C:\\Users\\w946225.GP0000231812\\Coureses_Haipeng_Tang\\CSC 611\\Project2\\dementia_prediciton\\MainProject\\project\\data\\test_rf.csv'
#path7='data\\train1.csv'
#path8='data\\test1.csv'
#path9='data1\\train3.csv'
#path10='data1\\test3.csv'

data_train=pd.read_csv(path5)
#data_train=data_train.fillna(data_train.mean())
data_test=pd.read_csv(path6)
#data_test=data_test.fillna(data_test.mean())

#查看类别分布情况
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False   #用来正常显示负号

print(data_train['Group'].value_counts())
data_train.Group.value_counts().plot(kind='bar')
plt.title(u'训练集')
plt.ylabel(u'人数')
plt.show()

print(data_test['Group'].value_counts())
data_train.Group.value_counts().plot(kind='bar')
plt.title(u'测试集')
plt.ylabel(u'人数')
plt.show()


# In[2]:


X1 = data_train.drop("Group", axis=1) #获取训练集特征  
y1 = data_train["Group"]  #获取训练集标签列 
X2 = data_test.drop("Group", axis=1) #获取测试集特征  
y2 = data_test["Group"]  #获取测试集标签列


# In[3]:


#模型训练
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve

random_state = 2018

def get_best_model(X1,y1,random_state):

    cv = StratifiedKFold(n_splits =5, shuffle=True, random_state = random_state)

    #训练朴素贝叶斯模型
    bys = BernoulliNB()
    parameters={
         'alpha':[1.0,2.0,3.0],
         'binarize':[0.0,None,1.0,2.0,3.0],
         'fit_prior':[True,False]
     }

     

    grid_clf = GridSearchCV(estimator = bys, param_grid = parameters, cv = cv, n_jobs=-1, verbose=4)
    grid_clf.fit(X1, y1)
    return grid_clf   
grid_clf = get_best_model(X1,y1,random_state)
print(grid_clf.best_estimator_)
print(grid_clf.best_params_)


# In[4]:


##得到的最优模型评估训练集
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import cross_val_predict

grid_clf = grid_clf.best_estimator_
target_name = ['Normal','MCI','VMD','dementia']
predicted = cross_val_predict(grid_clf, X1, y1, cv=5)
metrics.accuracy_score(y1, predicted) 
#朴素贝叶斯模型评估结果 加权重结果类别不平衡问题
print('naive bayes model准确度分数:',accuracy_score(y1, predicted)) #准确度分类得分
print('naive bayes model混淆矩阵\n',confusion_matrix(y1,predicted)) #混淆矩阵
print("naive bayes model召回率",recall_score(y1,predicted,average='macro'))#宏查全率
print("naive bayes model分类报告\n",classification_report(y1,predicted,target_names = target_name))#分类报告


# In[5]:


#得到的最优模型评估测试集
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve

#grid_clf = grid_clf.best_estimator_
target_name = ['Normal','MCI','VMD','dementia']


def get_RF_score(grid_clf,target_name):
    
    y_pred_RFC= grid_clf.fit(X1,y1).predict(X2)
    
    #朴素贝叶斯模型评估结果 加权重结果类别不平衡问题
    print('naive bayes model准确度分数:',accuracy_score(y2, y_pred_RFC)) #准确度分类得分
    print('naive bayes model混淆矩阵\n',confusion_matrix(y2,y_pred_RFC)) #混淆矩阵
    print("naive bayes model召回率",recall_score(y2,y_pred_RFC,average='macro'))#宏查全率
    print("naive bayes model分类报告\n",classification_report(y2,y_pred_RFC,target_names = target_name))#分类报告

    
get_RF_score(grid_clf,target_name)


# In[6]:


get_ipython().run_line_magic('pinfo', 'BernoulliNB')

