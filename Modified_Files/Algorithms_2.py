#!/usr/bin/env python
# coding: utf-8

# ###### Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import time

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap

#sklearn
from sklearn import datasets, svm, metrics,tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#preprocessing
from sklearn.preprocessing import StandardScaler,normalize
# Dimenionality Reduction
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import random_projection
#Feature selection
from sklearn.feature_selection import VarianceThreshold
#Under sampling
from imblearn.under_sampling import RandomUnderSampler
#Over sampling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,RandomOverSampler
#Combined sampling
from imblearn.combine import SMOTETomek
#Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier,AdaBoostClassifier,RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier,Perceptron,PassiveAggressiveClassifier,RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import resample
from sklearn.pipeline import *
from sklearn import metrics
from sklearn.metrics import f1_score,confusion_matrix,classification_report,make_scorer,average_precision_score,precision_recall_curve

#import pandas_ml as pdml

import warnings
warnings.filterwarnings("ignore")

from scipy.stats import itemfreq
import importlib
from importlib import reload  
from collections import defaultdict,Counter

from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.display.max_columns=200


from sklearn.metrics import accuracy_score,f1_score


np.random.seed(42)


# ###### Importing preprocessing file

# In[2]:


import Preprocessing as pp

#reload(pp)


# ###### Splitting target variable from other variables

# In[3]:


df=pp.cvp_ohe
train_data=df.copy(deep=True)
train_data=train_data.drop(['PRIM_CONTRIBUTORY_CAUSE', 'SEC_CONTRIBUTORY_CAUSE'], axis=1)
train_labels=df[['PRIM_CONTRIBUTORY_CAUSE']].copy()


# ###### Splitting to train and test 

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels,test_size=0.20,stratify=train_labels,random_state=42)


# ###### Scaling of data

# In[5]:


scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# ### Algorithms Implementation

# ##### Decision Tree Classifier

# ###### DecisionTreeClassifier - Approach 1

# In[6]:


feature_selection=PCA(n_components=500)
clf=tree.DecisionTreeClassifier()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_DT1 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for DecisionTreeClassifier - Approach 1

# In[7]:


print('********Decision Tree Classifier, Standard Scaled, PCA=500','********')
f1_score_micro_DT1 = metrics.f1_score(y_test, y_pred_DT1, average='micro')
f1_score_wt_DT1 = metrics.f1_score(y_test,y_pred_DT1,average='weighted')
precision_score_wt_DT1=metrics.precision_score(y_test, y_pred_DT1, average='weighted')
recall_score_wt_DT1=metrics.recall_score(y_test, y_pred_DT1, average='weighted')
print('F1-score_micro = ',f1_score_micro_DT1)
print('F1-score = ',f1_score_wt_DT1)
print('Precision = ',precision_score_wt_DT1)
print('Recall Score = ',recall_score_wt_DT1)


# ###### DecisionTreeClassifier - Approach 2

# In[8]:


feature_selection=VarianceThreshold()
clf=tree.DecisionTreeClassifier()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_DT2 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for DecisionTreeClassifier - Approach 2

# In[9]:


print('********Decision Tree Classifier, Standard Scaled, VarianceThreshold','********')
f1_score_micro_DT2 = metrics.f1_score(y_test, y_pred_DT2, average='micro')
f1_score_wt_DT2 = metrics.f1_score(y_test,y_pred_DT2,average='weighted')
precision_score_wt_DT2=metrics.precision_score(y_test, y_pred_DT2, average='weighted')
recall_score_wt_DT2=metrics.recall_score(y_test, y_pred_DT2, average='weighted')

print('F1-score_micro = ',f1_score_micro_DT2)
print('F1-score = ',f1_score_wt_DT2)
print('Precision = ',precision_score_wt_DT2)
print('Recall Score = ',recall_score_wt_DT2)


# ###### DecisionTreeClassifier - Approach 3

# In[10]:


feature_selection=TruncatedSVD(n_components=500)
clf=tree.DecisionTreeClassifier()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_DT3 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for DecisionTreeClassifier - Approach 3

# In[11]:


print('********Decision Tree Classifier, Standard Scaled, TruncatedSVD=500','********')
f1_score_micro_DT3 = metrics.f1_score(y_test, y_pred_DT3, average='micro')
f1_score_wt_DT3 = metrics.f1_score(y_test,y_pred_DT3,average='weighted')
precision_score_wt_DT3=metrics.precision_score(y_test, y_pred_DT3, average='weighted')
recall_score_wt_DT3=metrics.recall_score(y_test, y_pred_DT3, average='weighted')

print('F1-score_micro = ',f1_score_micro_DT3)
print('F1-score = ',f1_score_wt_DT3)
print('Precision = ',precision_score_wt_DT3)
print('Recall Score = ',recall_score_wt_DT3)


# In[ ]:





# ## LDA - LinearDiscriminantAnalysis

# ###### LDA - Approach 1

# In[12]:


feature_selection=PCA(n_components=500)
clf=LDA()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_LDA1 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for LDA - Approach 1

# In[13]:


print('********LDA, Standard Scaled, PCA=500','********')
f1_score_micro_LDA1 = metrics.f1_score(y_test, y_pred_LDA1, average='micro')
f1_score_wt_LDA1 = metrics.f1_score(y_test,y_pred_LDA1,average='weighted')
precision_score_wt_LDA1=metrics.precision_score(y_test, y_pred_LDA1, average='weighted')
recall_score_wt_LDA1=metrics.recall_score(y_test, y_pred_LDA1, average='weighted')
print('F1-score_micro = ',f1_score_micro_LDA1)
print('F1-score = ',f1_score_wt_LDA1)
print('Precision = ',precision_score_wt_LDA1)
print('Recall Score = ',recall_score_wt_LDA1)


# ###### LDA - Approach 2

# In[14]:


feature_selection=VarianceThreshold()
clf=LDA()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_LDA2 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for LDA - Approach 2

# In[15]:


print('********LDA Classifier, Standard Scaled, VarianceThreshold','********')
f1_score_micro_LDA2 = metrics.f1_score(y_test, y_pred_LDA2, average='micro')
f1_score_wt_LDA2 = metrics.f1_score(y_test,y_pred_LDA2,average='weighted')
precision_score_wt_LDA2=metrics.precision_score(y_test, y_pred_LDA2, average='weighted')
recall_score_wt_LDA2=metrics.recall_score(y_test, y_pred_LDA2, average='weighted')

print('F1-score_micro = ',f1_score_micro_LDA2)
print('F1-score = ',f1_score_wt_LDA2)
print('Precision = ',precision_score_wt_LDA2)
print('Recall Score = ',recall_score_wt_LDA2)


# ###### LDA - Approach 3

# In[16]:


feature_selection=TruncatedSVD(n_components=500)
clf=LDA()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_LDA3 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for LDA - Approach 3

# In[17]:


print('********LDA, Standard Scaled, TruncatedSVD=500','********')
f1_score_micro_LDA3 = metrics.f1_score(y_test, y_pred_LDA3, average='micro')
f1_score_wt_LDA3 = metrics.f1_score(y_test,y_pred_LDA3,average='weighted')
precision_score_wt_LDA3=metrics.precision_score(y_test, y_pred_LDA3, average='weighted')
recall_score_wt_LDA3=metrics.recall_score(y_test, y_pred_LDA3, average='weighted')

print('F1-score_micro = ',f1_score_micro_LDA3)
print('F1-score = ',f1_score_wt_LDA3)
print('Precision = ',precision_score_wt_LDA3)
print('Recall Score = ',recall_score_wt_LDA3)


# 

# ## Extra Tree Classifier

# ###### ExtraTreeClassifier - Approach 1

# In[18]:


feature_selection=PCA(n_components=500)
clf=ExtraTreesClassifier()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_ET1 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for ExtraTreeClassifier - Approach 1

# In[19]:


print('********Extra Tree Classifier, Standard Scaled, PCA=500','********')
f1_score_micro_ET1 = metrics.f1_score(y_test, y_pred_ET1, average='micro')
f1_score_wt_ET1 = metrics.f1_score(y_test,y_pred_ET1,average='weighted')
precision_score_wt_ET1=metrics.precision_score(y_test, y_pred_ET1, average='weighted')
recall_score_wt_ET1=metrics.recall_score(y_test, y_pred_ET1, average='weighted')
print('F1-score_micro = ',f1_score_micro_ET1)
print('F1-score = ',f1_score_wt_ET1)
print('Precision = ',precision_score_wt_ET1)
print('Recall Score = ',recall_score_wt_ET1)


# ###### ExtraTree - Approach 2
# 

# In[20]:



feature_selection=VarianceThreshold()
clf=ExtraTreesClassifier()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_ET2 = pipe_clf.predict(X_test_scaled)




# ###### Performance Evaluation for ExtraTreeClassifier - Approach 2
# 
# 
# 

# In[21]:



print('********Extra Tree Classifier, Standard Scaled,  VarianceThreshold','********')
f1_score_micro_ET2 = metrics.f1_score(y_test, y_pred_ET2, average='micro')
f1_score_wt_ET2 = metrics.f1_score(y_test,y_pred_ET2,average='weighted')
precision_score_wt_ET2=metrics.precision_score(y_test, y_pred_ET2, average='weighted')
recall_score_wt_ET2=metrics.recall_score(y_test, y_pred_ET2, average='weighted')

print('F1-score_micro = ',f1_score_micro_ET2)
print('F1-score = ',f1_score_wt_ET2)
print('Precision = ',precision_score_wt_ET2)
print('Recall Score = ',recall_score_wt_ET2)


# ###### ExtraTreeClassifier - Approach 3
# 

# In[22]:



feature_selection=TruncatedSVD(n_components=500)
clf=ExtraTreesClassifier()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_ET3 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for ExtraTreeClassifier - Approach 3
# 

# In[23]:




print('********Extra Tree Classifier, Standard Scaled, TruncatedSVD','********')
f1_score_micro_ET3 = metrics.f1_score(y_test, y_pred_ET3, average='micro')
f1_score_wt_ET3 = metrics.f1_score(y_test,y_pred_ET3,average='weighted')
precision_score_wt_ET3=metrics.precision_score(y_test, y_pred_ET3, average='weighted')
recall_score_wt_ET3=metrics.recall_score(y_test, y_pred_ET3, average='weighted')

print('F1-score_micro = ',f1_score_micro_ET3)
print('F1-score = ',f1_score_wt_ET3)
print('Precision = ',precision_score_wt_ET3)
print('Recall Score = ',recall_score_wt_ET3)


# In[ ]:





# In[ ]:





# In[ ]:


'f1_score_micro_DT1':f1_score_micro_DT1,
'f1_score_micro_DT2':f1_score_micro_DT2,
'f1_score_micro_DT3':f1_score_micro_DT3,
'f1_score_micro_ET1':f1_score_micro_ET1,
'f1_score_micro_ET2':f1_score_micro_ET2,
'f1_score_micro_ET3':f1_score_micro_ET3,
'f1_score_micro_ET3':f1_score_micro_ET3,
'f1_score_micro_LDA1':f1_score_micro_LDA1,
'f1_score_micro_LDA2':f1_score_micro_LDA2,
'f1_score_micro_LDA3':f1_score_micro_LDA3,
    
'f1_score_wt_DT1':f1_score_wt_DT1,
'f1_score_wt_DT2':f1_score_wt_DT2,
'f1_score_wt_DT3':f1_score_wt_DT3,
'f1_score_wt_ET1':f1_score_wt_ET1,
'f1_score_wt_ET2':f1_score_wt_ET2,
'f1_score_wt_ET3':f1_score_wt_ET3,
'f1_score_wt_ET3':f1_score_wt_ET3,
'f1_score_wt_LDA1':f1_score_wt_LDA1,
'f1_score_wt_LDA2':f1_score_wt_LDA2,
'f1_score_wt_LDA3':f1_score_wt_LDA3,
    
'precision_score_wt_DT1':precision_score_wt_DT1,
'precision_score_wt_DT2':precision_score_wt_DT2,
'precision_score_wt_DT3':precision_score_wt_DT3,
'precision_score_wt_ET1':precision_score_wt_ET1,
'precision_score_wt_ET2':precision_score_wt_ET2,
'precision_score_wt_ET3':precision_score_wt_ET3,
'precision_score_wt_ET3':precision_score_wt_ET3,
'precision_score_wt_LDA1':precision_score_wt_LDA1,
'precision_score_wt_LDA2':precision_score_wt_LDA2,
'precision_score_wt_LDA3':precision_score_wt_LDA3,
'recall_score_wt_DT1':recall_score_wt_DT1,
'recall_score_wt_DT2':recall_score_wt_DT2,
'recall_score_wt_DT3':recall_score_wt_DT3,
'recall_score_wt_ET1':recall_score_wt_ET1,
'recall_score_wt_ET2':recall_score_wt_ET2,
'recall_score_wt_ET3':recall_score_wt_ET3,
'recall_score_wt_ET3':recall_score_wt_ET3,
'recall_score_wt_LDA1':recall_score_wt_LDA1,
'recall_score_wt_LDA2':recall_score_wt_LDA2,
'recall_score_wt_LDA3':recall_score_wt_LDA3


# In[ ]:





# In[36]:


import pickle


# In[38]:


algorithm_2 = {'f1_score_micro_DT1':f1_score_micro_DT1,
'f1_score_micro_DT2':f1_score_micro_DT2,
'f1_score_micro_DT3':f1_score_micro_DT3,
'f1_score_micro_ET1':f1_score_micro_ET1,
'f1_score_micro_ET2':f1_score_micro_ET2,
'f1_score_micro_ET3':f1_score_micro_ET3,
'f1_score_micro_ET3':f1_score_micro_ET3,
'f1_score_micro_LDA1':f1_score_micro_LDA1,
'f1_score_micro_LDA2':f1_score_micro_LDA2,
'f1_score_micro_LDA3':f1_score_micro_LDA3,
    
'f1_score_wt_DT1':f1_score_wt_DT1,
'f1_score_wt_DT2':f1_score_wt_DT2,
'f1_score_wt_DT3':f1_score_wt_DT3,
'f1_score_wt_ET1':f1_score_wt_ET1,
'f1_score_wt_ET2':f1_score_wt_ET2,
'f1_score_wt_ET3':f1_score_wt_ET3,
'f1_score_wt_ET3':f1_score_wt_ET3,
'f1_score_wt_LDA1':f1_score_wt_LDA1,
'f1_score_wt_LDA2':f1_score_wt_LDA2,
'f1_score_wt_LDA3':f1_score_wt_LDA3,
    
'precision_score_wt_DT1':precision_score_wt_DT1,
'precision_score_wt_DT2':precision_score_wt_DT2,
'precision_score_wt_DT3':precision_score_wt_DT3,
'precision_score_wt_ET1':precision_score_wt_ET1,
'precision_score_wt_ET2':precision_score_wt_ET2,
'precision_score_wt_ET3':precision_score_wt_ET3,
'precision_score_wt_ET3':precision_score_wt_ET3,
'precision_score_wt_LDA1':precision_score_wt_LDA1,
'precision_score_wt_LDA2':precision_score_wt_LDA2,
'precision_score_wt_LDA3':precision_score_wt_LDA3,
'recall_score_wt_DT1':recall_score_wt_DT1,
'recall_score_wt_DT2':recall_score_wt_DT2,
'recall_score_wt_DT3':recall_score_wt_DT3,
'recall_score_wt_ET1':recall_score_wt_ET1,
'recall_score_wt_ET2':recall_score_wt_ET2,
'recall_score_wt_ET3':recall_score_wt_ET3,
'recall_score_wt_ET3':recall_score_wt_ET3,
'recall_score_wt_LDA1':recall_score_wt_LDA1,
'recall_score_wt_LDA2':recall_score_wt_LDA2,
'recall_score_wt_LDA3':recall_score_wt_LDA3}
pickle.dump(algorithm_2,open("algorithm_2.p","wb"))


# In[ ]:





# In[ ]:




