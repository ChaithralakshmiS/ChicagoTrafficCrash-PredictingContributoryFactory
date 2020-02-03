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
#from imblearn.under_sampling import RandomUnderSampler
#Over sampling
#from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,RandomOverSampler
#Combined sampling
#from imblearn.combine import SMOTETomek
#Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier,AdaBoostClassifier,RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,GradientBoostingRegressor
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
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

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

#Graphs
get_ipython().run_line_magic('matplotlib', '')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#import seaborn as sns;
import pickle


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


# In[4]:


#ax = sns.heatmap(train_labels, vmin=0, vmax=1)


# ###### Splitting to train and test 
# ###### Scaling of data

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels,test_size=0.20,stratify=train_labels,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# ### Algorithms Implementation

# # Gaussian Naive Bayes - Approach 1

# In[6]:


feature_selection=PCA(n_components=500)
clf=GaussianNB()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_GNB1 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for Gaussian Naive Bayes - Approach 1

# In[7]:


print('********Gaussian Naive Bayes, Standard Scaled, PCA=500','********')
f1_score_micro_1_GNB1 = metrics.f1_score(y_test, y_pred_GNB1, average='micro')
f1_score_wt_1_GNB1 = metrics.f1_score(y_test,y_pred_GNB1,average='weighted')
precision_score_wt_1_GNB1 =metrics.precision_score(y_test, y_pred_GNB1, average='weighted')
recall_score_wt_1_GNB1 =metrics.recall_score(y_test, y_pred_GNB1, average='weighted')

print('F1-score_micro = ',f1_score_micro_1_GNB1)
print('F1-score = ',f1_score_wt_1_GNB1)
print('Precision = ',precision_score_wt_1_GNB1)
print('Recall Score = ',recall_score_wt_1_GNB1)


# # Gaussian Naive Bayes - Approach 2

# In[8]:


feature_selection=VarianceThreshold()
clf=GaussianNB()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_GNB2 = pipe_clf.predict(X_test_scaled)


# ###### Performance Evaluation for Gaussian Naive Bayes - Approach 2

# In[9]:


print('********Gaussian Naive Bayes, Standard Scaled, Variance threshold','********')

f1_score_micro_2_GNB2 = metrics.f1_score(y_test, y_pred_GNB2, average='micro')
f1_score_wt_2_GNB2 = metrics.f1_score(y_test,y_pred_GNB2,average='weighted')
precision_score_wt_2_GNB2 = metrics.precision_score(y_test, y_pred_GNB2, average='weighted')
recall_score_wt_2_GNB2 =metrics.recall_score(y_test, y_pred_GNB2, average='weighted')

print('F1-score_micro = ',f1_score_micro_2_GNB2)
print('F1-score = ',f1_score_wt_2_GNB2)
print('Precision = ',precision_score_wt_2_GNB2)
print('Recall Score = ',recall_score_wt_2_GNB2)


# # Gaussian Naive Bayes - Approach 3

# In[10]:


feature_selection=TruncatedSVD(n_components=500)
clf=GaussianNB()

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)

y_pred_GNB3 = pipe_clf.predict(X_test_scaled)


# #### Performance Evaluation for Gaussian Naive Bayes - Approach 3

# In[12]:


print('********Gaussian Naive Bayes, Standard Scaled, Truncated SVD','********')

f1_score_micro_2_GNB3 = metrics.f1_score(y_test, y_pred_GNB3, average='micro')
f1_score_wt_2_GNB3 = metrics.f1_score(y_test,y_pred_GNB3,average='weighted')
precision_score_wt_2_GNB3 = metrics.precision_score(y_test, y_pred_GNB3, average='weighted')
recall_score_wt_2_GNB3 = metrics.recall_score(y_test, y_pred_GNB3, average='weighted')

print('F1-score_micro = ',f1_score_micro_2_GNB3)
print('F1-score = ',f1_score_wt_2_GNB3)
print('Precision = ',precision_score_wt_2_GNB3)
print('Recall Score = ',recall_score_wt_2_GNB3)


# # Algorithm - II 

# # Logistic Regression - Approach 1

# In[13]:


feature_selection=PCA(n_components=500)
clf=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_LR1 = pipe_clf.predict(X_test_scaled)


# #### Performance Evaluation for Logistic Regression - Approach 1

# In[14]:


print('********Logistic Regression, Standard Scaled, PCA=500','********')
f1_score_micro_1_LR1 = metrics.f1_score(y_test, y_pred_LR1, average='micro')
f1_score_wt_1_LR1 = metrics.f1_score(y_test,y_pred_LR1,average='weighted')
precision_score_wt_1_LR1 = metrics.precision_score(y_test, y_pred_LR1, average='weighted')
recall_score_wt_1_LR1 = metrics.recall_score(y_test, y_pred_LR1, average='weighted')

print('F1-score_micro = ',f1_score_micro_1_LR1)
print('F1-score = ',f1_score_wt_1_LR1)
print('Precision = ',precision_score_wt_1_LR1)
print('Recall Score = ',recall_score_wt_1_LR1)


# # Logistic Regression - Approach 2

# In[15]:


feature_selection=VarianceThreshold()
clf=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_LR2 = pipe_clf.predict(X_test_scaled)


# #### Performance Evaluation for Logistic Regression - Approach 2

# In[16]:


print('********Logistic Regression, Standard Scaled, Variance threshold','********')

f1_score_micro_2_LR2 = metrics.f1_score(y_test, y_pred_LR2, average='micro')
f1_score_wt_2_LR2 = metrics.f1_score(y_test,y_pred_LR2,average='weighted')
precision_score_wt_2_LR2 = metrics.precision_score(y_test, y_pred_LR2, average='weighted')
recall_score_wt_2_LR2 = metrics.recall_score(y_test, y_pred_LR2, average='weighted')

print('F1-score_micro = ',f1_score_micro_2_LR2)
print('F1-score = ',f1_score_wt_2_LR2)
print('Precision = ',precision_score_wt_2_LR2)
print('Recall Score = ',recall_score_wt_2_LR2)


# # Logistic Regression - Approach 3

# In[17]:


feature_selection=TruncatedSVD(n_components=500)
clf=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)

y_pred_LR3 = pipe_clf.predict(X_test_scaled)


# #### Performance Evaluation for Logistic Regression - Approach 3

# In[18]:


print('********Logistic Regression, Standard Scaled, Truncated SVD','********')

f1_score_tsvd_micro_LR3 = metrics.f1_score(y_test, y_pred_LR3, average='micro')
f1_score_tsvd_wt_LR3 = metrics.f1_score(y_test,y_pred_LR3,average='weighted')
precision_score_tsvd_wt_LR3 = metrics.precision_score(y_test, y_pred_LR3, average='weighted')
recall_score_tsvd_wt_LR3 = metrics.recall_score(y_test, y_pred_LR3, average='weighted')

print('F1-score_micro = ',f1_score_tsvd_micro_LR3)
print('F1-score = ',f1_score_tsvd_wt_LR3)
print('Precision = ',precision_score_tsvd_wt_LR3)
print('Recall Score = ',recall_score_tsvd_wt_LR3)


# # Algorithm III

# # Random Forest - Approach 1

# In[19]:


feature_selection=PCA(n_components=500)
clf=RandomForestClassifier(n_estimators=100, random_state=100)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_RF1 = pipe_clf.predict(X_test_scaled)


# #### Performance Evaluation for Random Forest - Approach 1

# In[20]:


print('********Random Forest, Standard Scaled, PCA=500','********')
f1_score_micro_1_RF1 = metrics.f1_score(y_test, y_pred_RF1, average='micro')
f1_score_wt_1_RF1 = metrics.f1_score(y_test,y_pred_RF1,average='weighted')
precision_score_wt_1_RF1 = metrics.precision_score(y_test, y_pred_RF1, average='weighted')
recall_score_wt_1_RF1 = metrics.recall_score(y_test, y_pred_RF1, average='weighted')

print('F1-score_micro = ',f1_score_micro_1_RF1)
print('F1-score = ',f1_score_wt_1_RF1)
print('Precision = ',precision_score_wt_1_RF1)
print('Recall Score = ',recall_score_wt_1_RF1)


# # Random Forest - Approach 2

# In[21]:


feature_selection=VarianceThreshold()
clf=RandomForestClassifier(n_estimators=100, random_state=100)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_RF2 = pipe_clf.predict(X_test_scaled)


# #### Performance Evaluation for Random Forest - Approach 2

# In[22]:


print('********Random Forest, Standard Scaled, Variance threshold','********')

f1_score_micro_2_RF2 = metrics.f1_score(y_test, y_pred_RF2, average='micro')
f1_score_wt_2_RF2 = metrics.f1_score(y_test,y_pred_RF2,average='weighted')
precision_score_wt_2_RF2 = metrics.precision_score(y_test, y_pred_RF2, average='weighted')
recall_score_wt_2_RF2 = metrics.recall_score(y_test, y_pred_RF2, average='weighted')

print('F1-score_micro = ',f1_score_micro_2_RF2)
print('F1-score = ',f1_score_wt_2_RF2)
print('Precision = ',precision_score_wt_2_RF2)
print('Recall Score = ',recall_score_wt_2_RF2)


# # Random Forest - Approach 3

# In[23]:


feature_selection=TruncatedSVD(n_components=500)
clf=RandomForestClassifier(n_estimators=100, random_state=100)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)

y_pred_RF3 = pipe_clf.predict(X_test_scaled)


# #### Performance Evaluation for Random Forest - Approach 3

# In[24]:


print('********Random Forest, Standard Scaled, Variance threshold','********')

f1_score_micro_2_RF3 = metrics.f1_score(y_test, y_pred_RF3, average='micro')
f1_score_wt_2_RF3 = metrics.f1_score(y_test,y_pred_RF3,average='weighted')
precision_score_wt_2_RF3 = metrics.precision_score(y_test, y_pred_RF3, average='weighted')
recall_score_wt_2_RF3 = metrics.recall_score(y_test, y_pred_RF3, average='weighted')

print('F1-score_micro = ',f1_score_micro_2_RF3)
print('F1-score = ',f1_score_wt_2_RF3)
print('Precision = ',precision_score_wt_2_RF3)
print('Recall Score = ',recall_score_wt_2_RF3)


# # Random Forest - Approach 4

# In[25]:


feature_selection=PCA(n_components=500)
clf=RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=100, max_features=10)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_RF4 = pipe_clf.predict(X_test_scaled)


# #### Performance Evaluation for Random Forest - Approach 4

# In[26]:


print('********Random Forest, Standard Scaled, PCA=500','********')

f1_score_micro_2_RF4 = metrics.f1_score(y_test, y_pred_RF4, average='micro')
f1_score_wt_2_RF4 = metrics.f1_score(y_test,y_pred_RF4,average='weighted')
precision_score_wt_2_RF4 = metrics.precision_score(y_test, y_pred_RF4, average='weighted')
recall_score_wt_2_RF4 = metrics.recall_score(y_test, y_pred_RF4, average='weighted')

print('F1-score_micro = ',f1_score_micro_2_RF4)
print('F1-score = ',f1_score_wt_2_RF4)
print('Precision = ',precision_score_wt_2_RF4)
print('Recall Score = ',recall_score_wt_2_RF4)


# In[28]:


import pickle
algorithm_1 = {'f1_score_micro_1_GNB1':f1_score_micro_1_GNB1,
'f1_score_wt_1_GNB1': f1_score_wt_1_GNB1,
'precision_score_wt_1_GNB1': precision_score_wt_1_GNB1,
'recall_score_wt_1_GNB1':recall_score_wt_1_GNB1,
'f1_score_micro_2_GNB2':f1_score_micro_2_GNB2,
'f1_score_wt_2_GNB2':f1_score_wt_2_GNB2,
'precision_score_wt_2_GNB2':precision_score_wt_2_GNB2,
'recall_score_wt_2_GNB2':recall_score_wt_2_GNB2,
'f1_score_micro_2_GNB3':f1_score_micro_2_GNB3,
'f1_score_wt_2_GNB3':f1_score_wt_2_GNB3,
'precision_score_wt_2_GNB3':precision_score_wt_2_GNB3,
'recall_score_wt_2_GNB3':recall_score_wt_2_GNB3,
'f1_score_micro_1_LR1':f1_score_micro_1_LR1,
'f1_score_wt_1_LR1':f1_score_wt_1_LR1,
'precision_score_wt_1_LR1':precision_score_wt_1_LR1,
'recall_score_wt_1_LR1':recall_score_wt_1_LR1,
'f1_score_micro_2_LR2':f1_score_micro_2_LR2,
'f1_score_wt_2_LR2':f1_score_wt_2_LR2,
'precision_score_wt_2_LR2':precision_score_wt_2_LR2,
'recall_score_wt_2_LR2':recall_score_wt_2_LR2,
'f1_score_tsvd_micro_LR3':f1_score_tsvd_micro_LR3,
'f1_score_tsvd_wt_LR3':f1_score_tsvd_wt_LR3,
'precision_score_tsvd_wt_LR3':precision_score_tsvd_wt_LR3,
'recall_score_tsvd_wt_LR3':recall_score_tsvd_wt_LR3,
'f1_score_micro_1_RF1':f1_score_micro_1_RF1,
'f1_score_wt_1_RF1':f1_score_wt_1_RF1,
'precision_score_wt_1_RF1':precision_score_wt_1_RF1,
'recall_score_wt_1_RF1':recall_score_wt_1_RF1,
'f1_score_micro_2_RF2':f1_score_micro_2_RF2,
'f1_score_wt_2_RF2':f1_score_wt_2_RF2,
'precision_score_wt_2_RF2':precision_score_wt_2_RF2,
'recall_score_wt_2_RF2':recall_score_wt_2_RF2,
'f1_score_micro_2_RF3':f1_score_micro_2_RF3,
'f1_score_wt_2_RF3':f1_score_wt_2_RF3,
'precision_score_wt_2_RF3':precision_score_wt_2_RF3,
'recall_score_wt_2_RF3':recall_score_wt_2_RF3,
'f1_score_micro_2_RF4':f1_score_micro_2_RF4,
'f1_score_wt_2_RF4':f1_score_wt_2_RF4,
'precision_score_wt_2_RF4':precision_score_wt_2_RF4,
'recall_score_wt_2_RF4':recall_score_wt_2_RF4
}

pickle.dump(algorithm_1,open("algorithm_1.p","wb"))


# In[ ]:




