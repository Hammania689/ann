
# coding: utf-8

# In[1]:


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve

random_state = 2018


def get_best_model(X_train,y_train,lrc,random_state):

    cv = StratifiedKFold(n_splits =5, shuffle=True, random_state = random_state)

    #训练神经网络模型
    mlp = MLPClassifier(random_state = random_state) 
    scoring = {'Recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score)
              }

    params = { 'hidden_layer_sizes':[10,50,100,200,500],
              'activation':['identity', 'logistic', 'tanh', 'relu'],
              'solver':['lbfgs', 'sgd', 'adam'],
              'learning_rate':['constant', 'invscaling', 'adaptive']
              
                 }

    grid_clf = GridSearchCV(estimator = mlp, param_grid = params, cv = cv, n_jobs=-1, verbose=4)
    grid_clf.fit(X_train, y_train)
    return grid_clf   
grid_clf = get_best_model(X_train,y_train,lrc,random_state)
best_mlp = grid_clf.best_estimator_
print(grid_clf.best_estimator_)
print(grid_clf.best_params_)


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc,             classification_report, recall_score, precision_recall_curve

def get_mlp_score_cross(clf):
    target_name = ['Normal','MCI','VMD','dementia']
    predicted = cross_val_predict(clf, X_train, y_train, cv=5)
    #LogisticRegression模型交叉验证评估结果 加权重结果类别不平衡问题
    print('MLP model准确度分数:',accuracy_score(y_train, predicted)) #准确度分类得分
    print('MLP model混淆矩阵\n',confusion_matrix(y_train,predicted)) #混淆矩阵
    print("MLP model召回率",recall_score(y_train,predicted,average='micro'))#宏查全率
    print("MLP model分类报告\n",classification_report(y_train,predicted,target_names = target_name))#分类报告

    
    
def get_mlp_score_test(clf):
    target_name = ['Normal','MCI','VMD','dementia']
    y_pred_score = clf.fit(X_train,y_train).predict(X_test)
    #LogisticRegression模型 测试集 评估结果 加权重结果类别不平衡问题
    print('MLP model准确度分数:',accuracy_score(y_test, y_pred_score)) #准确度分类得分
    print('MLP model混淆矩阵\n',confusion_matrix(y_test,y_pred_score)) #混淆矩阵
    print("MLP model召回率",recall_score(y_test,y_pred_score,average='micro'))#宏查全率
    print("MLP model分类报告\n",classification_report(y_test,y_pred_score,target_names = target_name))#分类报告


print('对训练集使用交叉验证得到的评估结果')
get_mlp_score_cross(best_mlp)
print('使用测试集得到的评估结果')
get_mlp_score_test(best_mlp)

