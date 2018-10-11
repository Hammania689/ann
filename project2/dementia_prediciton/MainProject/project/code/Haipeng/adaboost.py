
# coding: utf-8

# In[ ]:


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

data_train=pd.read_csv(path3)
#data_train=data_train.fillna(data_train.mean())
data_test=pd.read_csv(path4)
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


# In[5]:


X1 = data_train.drop("Group", axis=1) #获取训练集特征  
y1 = data_train["Group"]  #获取训练集标签列 
X2 = data_test.drop("Group", axis=1) #获取测试集特征  
y2 = data_test["Group"]  #获取测试集标签列


# In[6]:


#模型训练
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve

random_state = 2018

def get_best_model(X1,y1,random_state):

    cv = StratifiedKFold(n_splits =5, shuffle=True, random_state = random_state)

    #训练模型
    lrc = LogisticRegression(random_state = random_state) 
    scoring = {'Recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)
              }

    params = {'class_weight':['balanced',None],
                    'penalty':['l2'], 
                    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                    'multi_class' : ['ovr'],
                    'C':[0.0001,0.001,0.01,0.1,1,10]
                 }

    grid_clf = GridSearchCV(estimator = lrc, param_grid = params, cv = cv, n_jobs=-1, verbose=4)
    grid_clf.fit(X1, y1)
    return grid_clf   
grid_clf = get_best_model(X1,y1,random_state)
print(grid_clf.best_estimator_)
print(grid_clf.best_params_)


# In[35]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve

random_state = 2018

lrc = LogisticRegression(C=10, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=2018,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

def get_best_model(X1,y1,lrc,random_state):

    cv = StratifiedKFold(n_splits =5, shuffle=True, random_state = random_state)

    #训练Adaboost模型
    lrc = AdaBoostClassifier(base_estimator = lrc,random_state = random_state) 
    scoring = {'Recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)
              }

    params = { 'n_estimators':[10,30,50,70,100],
              'algorithm':['SAMME', 'SAMME.R']
                 }

    grid_clf = GridSearchCV(estimator = lrc, param_grid = params, cv = cv, n_jobs=-1, verbose=4)
    grid_clf.fit(X1, y1)
    return grid_clf   
grid_clf = get_best_model(X1,y1,lrc,random_state)
best_clf = grid_clf.best_estimator_
print(grid_clf.best_estimator_)
print(grid_clf.best_params_)


# In[36]:


best_clf = AdaBoostClassifier(algorithm='SAMME',
          base_estimator=LogisticRegression(C=10, class_weight='balanced', dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=2018,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
          learning_rate=1.0, n_estimators=10, random_state=2018)


# In[37]:


##得到的最优模型评估训练集
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve,accuracy_score

best_lrc = lrc
def get_lr_score_cross(clf):
    target_name = ['Normal','MCI','VMD','dementia']
    predicted = cross_val_predict(clf, X1, y1, cv=5)
    #LogisticRegression模型交叉验证评估结果 加权重结果类别不平衡问题
    print('Logistic model准确度分数:',accuracy_score(y1, predicted)) #准确度分类得分
    print('Logistic model混淆矩阵\n',confusion_matrix(y1,predicted)) #混淆矩阵
    print("Logistic model召回率",recall_score(y1,predicted,average='micro'))#宏查全率
    print("Logistic model分类报告\n",classification_report(y1,predicted,target_names = target_name))#分类报告

    
    
def get_lr_score_test(clf):
    target_name = ['Normal','MCI','VMD','dementia']
    y_pred_score = clf.fit(X1,y1).predict(X2)
    #LogisticRegression模型 测试集 评估结果 加权重结果类别不平衡问题
    print('Logistic model准确度分数:',accuracy_score(y2, y_pred_score)) #准确度分类得分
    print('Logistic model混淆矩阵\n',confusion_matrix(y2,y_pred_score)) #混淆矩阵
    print("Logistic model召回率",recall_score(y2,y_pred_score,average='micro'))#宏查全率
    print("Logistic model分类报告\n",classification_report(y2,y_pred_score,target_names = target_name))#分类报告


print("交叉验证得到的评估结果")
get_lr_score_cross(best_lrc)
print('使用测试集得到的评估结果')
get_lr_score_test(best_lrc)


# In[38]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve


best_boostclf = best_clf    
def get_lrboost_score_cross(clf):
    target_name = ['Normal','MCI','VMD','dementia']
    predicted = cross_val_predict(best_lrc, X1, y1, cv=5)
    #LogisticRegression模型交叉验证评估结果 加权重结果类别不平衡问题
    print('AdaBoost model准确度分数:',accuracy_score(y1, predicted)) #准确度分类得分
    print('AdaBoost model混淆矩阵\n',confusion_matrix(y1,predicted)) #混淆矩阵
    print("AdaBoost model召回率",recall_score(y1,predicted,average='micro'))#宏查全率
    print("AdaBoost model分类报告\n",classification_report(y1,predicted,target_names = target_name))#分类报告

    
    
def get_lrboost_score_test(clf):
    target_name = ['Normal','MCI','VMD','dementia']
    y_pred_score = clf.fit(X1,y1).predict(X2)
    #LogisticRegression模型 测试集 评估结果 加权重结果类别不平衡问题
    print('AdaBoost model准确度分数:',accuracy_score(y2, y_pred_score)) #准确度分类得分
    print('AdaBoost model混淆矩阵\n',confusion_matrix(y2,y_pred_score)) #混淆矩阵
    print("AdaBoost model召回率",recall_score(y2,y_pred_score,average='micro'))#宏查全率
    print("AdaBoost model分类报告\n",classification_report(y2,y_pred_score,target_names = target_name))#分类报告


print('对训练集使用交叉验证得到的评估结果')
get_lrboost_score_cross(best_boostclf)
print('使用测试集得到的评估结果')
get_lrboost_score_test(best_boostclf)


# In[39]:


get_ipython().run_line_magic('pinfo', 'AdaBoostClassifier')

