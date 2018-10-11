# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 09:43:10 2018

@author: Administrator
"""
# -*- coding: utf-8 -*-  
  
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestClassifier  
import pickle  
import matplotlib.pyplot as plt  

'''  
with open('training_df.pkl', 'rb') as f:  
    df = pickle.load(f)  
print("data loaded")  
''' 
#读入train.csv文件
#path='F:\\data\\train.csv'
path='C:\\Users\\w946225.GP0000231812\\Coureses_Haipeng_Tang\\CSC 611\\Project2\\dementia_prediciton\\MainProject\\project\\data'
df= pd.read_csv(path)
df=df.fillna(df.mean()) #填补缺失数据
y = df["Group"]                                   #获取标签列  
X = df.drop("Group", axis=1)                      #剩下的所有特征  
  
for i in range(10):                           #这里我们进行十次循环取交集  
    tmp = set()  
    rfc = RandomForestClassifier(n_jobs=-1)  
    rfc.fit(X, y)  
    print("training finished")  
  
    importances = rfc.feature_importances_  
    indices = np.argsort(importances)[::-1]   # 降序排列  
    for f in range(X.shape[1]):  
        if f < 50:                            #选出前50个重要的特征  
            tmp.add(X.columns[indices[f]]) 
        print("%2d) %-*s %f" % (f + 1, 30, X.columns[indices[f]], importances[indices[f]]))  
  
    selected_feat_names= tmp  
    print(len(selected_feat_names), "features are selected")  
  
plt.title("Feature Importance")  
plt.bar(range(X.shape[1]),  
        importances[indices],  
        color='lightblue',  
        align='center')  
plt.xticks(range(X.shape[1]),  
           X.columns[indices],  
           rotation=90)  
plt.xlim([-1, X.shape[1]])  
plt.tight_layout()  
plt.show()  
   
'''  
with open(r'selected_feat_names.pkl', 'wb') as f:  
    pickle.dump(list(selected_feat_names), f)  
'''
with open(r'randomforest730.pkl', 'wb') as f:  
    pickle.dump(list(selected_feat_names), f)  