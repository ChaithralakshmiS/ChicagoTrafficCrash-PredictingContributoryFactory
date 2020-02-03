
# coding: utf-8

# ###### Importing libraries

# In[34]:


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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
from sklearn.ensemble import VotingClassifier,AdaBoostClassifier,RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
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


# ###### Splitting to train and test / Scaling of data

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels,test_size=0.20,stratify=train_labels,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


# ### Algorithms Implementation

# ###### AdaBoostClassifier - Approach 1

# In[21]:


feature_selection=PCA(n_components=500)
clf = AdaBoostClassifier(n_estimators=500)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_AB1 = pipe_clf.predict(X_test_scaled)


# In[23]:


print('********AdaBoost Classifier, Standard Scaled, PCA=500','********')
f1_score_micro_AB1 = metrics.f1_score(y_test, y_pred_AB1, average='micro')
f1_score_wt_AB1 = metrics.f1_score(y_test,y_pred_AB1,average='weighted')
precision_score_wt_AB1=metrics.precision_score(y_test, y_pred_AB1, average='weighted')
recall_score_wt_AB1=metrics.recall_score(y_test, y_pred_AB1, average='weighted')

print('F1-score_micro = ',f1_score_micro_AB1)
print('F1-score = ',f1_score_wt_AB1)
print('Precision = ',precision_score_wt_AB1)
print('Recall Score = ',recall_score_wt_AB1)

get_ipython().run_line_magic('store', 'f1_score_micro_AB1')
get_ipython().run_line_magic('store', 'f1_score_wt_AB1')
get_ipython().run_line_magic('store', 'precision_score_wt_AB1')
get_ipython().run_line_magic('store', 'recall_score_wt_AB1')


# ###### AdaBoostClassifier - Approach 2

# In[ ]:


feature_selection=VarianceThreshold()
clf = AdaBoostClassifier(n_estimators=500)
pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_AB2 = pipe_clf.predict(X_test_scaled)


# In[13]:


print('********AdaBoost Classifier, Standard Scaled, Variance threshold','********')

f1_score_micro_AB2 = metrics.f1_score(y_test, y_pred_AB2, average='micro')
f1_score_wt_AB2 = metrics.f1_score(y_test,y_pred_AB2,average='weighted')
precision_score_wt_AB2=metrics.precision_score(y_test, y_pred_AB2, average='weighted')
recall_score_wt_AB2=metrics.recall_score(y_test, y_pred_AB2, average='weighted')

print('F1-score_micro = ',f1_score_micro_AB2)
print('F1-score = ',f1_score_wt_AB2)
print('Precision = ',precision_score_wt_AB2)
print('Recall Score = ',recall_score_wt_AB2)

get_ipython().run_line_magic('store', 'f1_score_micro_AB2')
get_ipython().run_line_magic('store', 'f1_score_wt_AB2')
get_ipython().run_line_magic('store', 'precision_score_wt_AB2')
get_ipython().run_line_magic('store', 'recall_score_wt_AB2')


# ###### AdaBoostClassifier - Approach 3

# In[14]:


feature_selection=TruncatedSVD(n_components=500)
clf = AdaBoostClassifier(n_estimators=500)
pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)

y_pred_AB3 = pipe_clf.predict(X_test_scaled)


# In[17]:


print('********AdaBoost Classifier, Standard Scaled, Truncated SVD','********')

f1_score_micro_AB3 = metrics.f1_score(y_test, y_pred_AB3, average='micro')
f1_score_wt_AB3 = metrics.f1_score(y_test,y_pred_AB3,average='weighted')
precision_score_wt_AB3 = metrics.precision_score(y_test, y_pred_AB3, average='weighted')
recall_score_wt_AB3 = metrics.recall_score(y_test, y_pred_AB3, average='weighted')

print('F1-score_micro = ',f1_score_micro_AB3)
print('F1-score = ',f1_score_wt_AB3)
print('Precision = ',precision_score_wt_AB3)
print('Recall Score = ',recall_score_wt_AB3)

get_ipython().run_line_magic('store', 'f1_score_micro_AB3')
get_ipython().run_line_magic('store', 'f1_score_wt_AB3')
get_ipython().run_line_magic('store', 'precision_score_wt_AB3')
get_ipython().run_line_magic('store', 'recall_score_wt_AB3')


# ###### MultiLayerPerceptron - Approach 1

# In[5]:


feature_selection=PCA(n_components=500)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_MLP1 = pipe_clf.predict(X_test_scaled)


# In[18]:


print('********MultiLayerPerceptron, Standard Scaled, PCA','********')

f1_score_micro_MLP1 = metrics.f1_score(y_test, y_pred_MLP1, average='micro')
f1_score_wt_MLP1 = metrics.f1_score(y_test,y_pred_MLP1,average='weighted')
precision_score_wt_MLP1 = metrics.precision_score(y_test, y_pred_MLP1, average='weighted')
recall_score_wt_MLP1 = metrics.recall_score(y_test, y_pred_MLP1, average='weighted')

print('F1-score_micro = ',f1_score_micro_MLP1)
print('F1-score = ',f1_score_wt_MLP1)
print('Precision = ',precision_score_wt_MLP1)
print('Recall Score = ',recall_score_wt_MLP1)

get_ipython().run_line_magic('store', 'f1_score_micro_MLP1')
get_ipython().run_line_magic('store', 'f1_score_wt_MLP1')
get_ipython().run_line_magic('store', 'precision_score_wt_MLP1')
get_ipython().run_line_magic('store', 'recall_score_wt_MLP1')


# ###### MultiLayerPerceptron - Approach 2

# In[7]:


feature_selection=VarianceThreshold()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_MLP2 = pipe_clf.predict(X_test_scaled)


# In[19]:


print('********MultiLayerPerceptron, Standard Scaled, Variance threshold','********')

f1_score_micro_MLP2 = metrics.f1_score(y_test, y_pred_MLP2, average='micro')
f1_score_wt_MLP2 = metrics.f1_score(y_test,y_pred_MLP2,average='weighted')
precision_score_wt_MLP2 = metrics.precision_score(y_test, y_pred_MLP2, average='weighted')
recall_score_wt_MLP2 = metrics.recall_score(y_test, y_pred_MLP2, average='weighted')

print('F1-score_micro = ',f1_score_micro_MLP2)
print('F1-score = ',f1_score_wt_MLP2)
print('Precision = ',precision_score_wt_MLP2)
print('Recall Score = ',recall_score_wt_MLP2)

get_ipython().run_line_magic('store', 'f1_score_micro_MLP2')
get_ipython().run_line_magic('store', 'f1_score_wt_MLP2')
get_ipython().run_line_magic('store', 'precision_score_wt_MLP2')
get_ipython().run_line_magic('store', 'recall_score_wt_MLP2')


# ###### MultiLayerPerceptron - Approach 3

# In[9]:


feature_selection=TruncatedSVD(n_components=500)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_MLP3 = pipe_clf.predict(X_test_scaled)


# In[20]:


print('********MultiLayerPerceptron, Standard Scaled, Variance threshold','********')

f1_score_micro_MLP3 = metrics.f1_score(y_test, y_pred_MLP3, average='micro')
f1_score_wt_MLP3 = metrics.f1_score(y_test,y_pred_MLP3,average='weighted')
precision_score_wt_MLP3 = metrics.precision_score(y_test, y_pred_MLP3, average='weighted')
recall_score_wt_MLP3 = metrics.recall_score(y_test, y_pred_MLP3, average='weighted')

print('F1-score_micro = ',f1_score_micro_MLP3)
print('F1-score = ',f1_score_wt_MLP3)
print('Precision = ',precision_score_wt_MLP3)
print('Recall Score = ',recall_score_wt_MLP3)

get_ipython().run_line_magic('store', 'f1_score_micro_MLP3')
get_ipython().run_line_magic('store', 'f1_score_wt_MLP3')
get_ipython().run_line_magic('store', 'precision_score_wt_MLP3')
get_ipython().run_line_magic('store', 'recall_score_wt_MLP3')


# ###### BaggingClassifierWithKNN - Approach 1

# In[26]:


feature_selection=PCA(n_components=500)
clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_BGKNN1 = pipe_clf.predict(X_test_scaled)


# In[27]:


print('********Bagging Classifier with KNN, Standard Scaled, PCA','********')

f1_score_micro_BGKNN1 = metrics.f1_score(y_test, y_pred_BGKNN1, average='micro')
f1_score_wt_BGKNN1 = metrics.f1_score(y_test,y_pred_BGKNN1,average='weighted')
precision_score_wt_BGKNN1 = metrics.precision_score(y_test, y_pred_BGKNN1, average='weighted')
recall_score_wt_BGKNN1 = metrics.recall_score(y_test, y_pred_BGKNN1, average='weighted')

print('F1-score_micro = ',f1_score_micro_BGKNN1)
print('F1-score = ',f1_score_wt_BGKNN1)
print('Precision = ',precision_score_wt_BGKNN1)
print('Recall Score = ',recall_score_wt_BGKNN1)

get_ipython().run_line_magic('store', 'f1_score_micro_BGKNN1')
get_ipython().run_line_magic('store', 'f1_score_wt_BGKNN1')
get_ipython().run_line_magic('store', 'precision_score_wt_BGKNN1')
get_ipython().run_line_magic('store', 'recall_score_wt_BGKNN1')
get_ipython().run_line_magic('store', 'f1_score_micro_BGET2')
get_ipython().run_line_magic('store', 'f1_score_wt_BGET2')
get_ipython().run_line_magic('store', 'precision_score_wt_BGET2')
get_ipython().run_line_magic('store', 'recall_score_wt_BGET2')
get_ipython().run_line_magic('store', 'f1_score_micro_BRF2')
get_ipython().run_line_magic('store', 'f1_score_wt_BRF2')
get_ipython().run_line_magic('store', 'precision_score_wt_BGRF2')
get_ipython().run_line_magic('store', 'recall_score_wt_BGRF2')
get_ipython().run_line_magic('store', 'f1_score_micro_BLDA2')
get_ipython().run_line_magic('store', 'f1_score_wt_BLDA2')
get_ipython().run_line_magic('store', 'precision_score_wt_BLDA2')
get_ipython().run_line_magic('store', 'recall_score_wt_BLDA2')
get_ipython().run_line_magic('store', 'f1_score_micro_DT2')
get_ipython().run_line_magic('store', 'f1_score_wt_DT2')
get_ipython().run_line_magic('store', 'precision_score_wt_DT2')
get_ipython().run_line_magic('store', 'recall_score_wt_DT2')


# ###### BaggingClassifierWith ExtraTrees

# In[29]:


feature_selection=VarianceThreshold()
clf = BaggingClassifier(ExtraTreesClassifier()) 

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_BGET2 = pipe_clf.predict(X_test_scaled)


# In[30]:


print('********Bagging Classifier with Extra Tree, Standard Scaled, VarianceThreshold','********')

f1_score_micro_BGET2 = metrics.f1_score(y_test, y_pred_BGET2, average='micro')
f1_score_wt_BGET2 = metrics.f1_score(y_test,y_pred_BGET2,average='weighted')
precision_score_wt_BGET2 = metrics.precision_score(y_test, y_pred_BGET2, average='weighted')
recall_score_wt_BGET2 = metrics.recall_score(y_test, y_pred_BGET2, average='weighted')

print('F1-score_micro = ',f1_score_micro_BGET2)
print('F1-score = ',f1_score_wt_BGET2)
print('Precision = ',precision_score_wt_BGET2)
print('Recall Score = ',recall_score_wt_BGET2)

get_ipython().run_line_magic('store', 'f1_score_micro_BGET2')
get_ipython().run_line_magic('store', 'f1_score_wt_BGET2')
get_ipython().run_line_magic('store', 'precision_score_wt_BGET2')
get_ipython().run_line_magic('store', 'recall_score_wt_BGET2')


# ###### BaggingClassifierWithRF

# In[31]:


feature_selection=VarianceThreshold()
clf = BaggingClassifier(RandomForestClassifier()) 

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_BRF2 = pipe_clf.predict(X_test_scaled)


# In[32]:


print('********Bagging Classifier with RF, Standard Scaled, Variance Threshold','********')

f1_score_micro_BRF2 = metrics.f1_score(y_test, y_pred_BRF2, average='micro')
f1_score_wt_BRF2 = metrics.f1_score(y_test,y_pred_BRF2,average='weighted')
precision_score_wt_BGRF2 = metrics.precision_score(y_test, y_pred_BRF2, average='weighted')
recall_score_wt_BGRF2 = metrics.recall_score(y_test, y_pred_BRF2, average='weighted')

print('F1-score_micro = ',f1_score_micro_BRF2)
print('F1-score = ',f1_score_wt_BRF2)
print('Precision = ',precision_score_wt_BGRF2)
print('Recall Score = ',recall_score_wt_BGRF2)

get_ipython().run_line_magic('store', 'f1_score_micro_BRF2')
get_ipython().run_line_magic('store', 'f1_score_wt_BRF2')
get_ipython().run_line_magic('store', 'precision_score_wt_BGRF2')
get_ipython().run_line_magic('store', 'recall_score_wt_BGRF2')


# ###### Bagging with LDA

# In[35]:


feature_selection=VarianceThreshold()
clf = BaggingClassifier(LinearDiscriminantAnalysis()) 

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_BLDA2 = pipe_clf.predict(X_test_scaled)


# In[36]:


print('********Bagging Classifier with LDA, Standard Scaled, Variance Threshold','********')

f1_score_micro_BLDA2 = metrics.f1_score(y_test, y_pred_BLDA2, average='micro')
f1_score_wt_BLDA2 = metrics.f1_score(y_test,y_pred_BLDA2,average='weighted')
precision_score_wt_BLDA2 = metrics.precision_score(y_test, y_pred_BLDA2, average='weighted')
recall_score_wt_BLDA2 = metrics.recall_score(y_test, y_pred_BLDA2, average='weighted')

print('F1-score_micro = ',f1_score_micro_BLDA2)
print('F1-score = ',f1_score_wt_BLDA2)
print('Precision = ',precision_score_wt_BLDA2)
print('Recall Score = ',recall_score_wt_BLDA2)

get_ipython().run_line_magic('store', 'f1_score_micro_BLDA2')
get_ipython().run_line_magic('store', 'f1_score_wt_BLDA2')
get_ipython().run_line_magic('store', 'precision_score_wt_BLDA2')
get_ipython().run_line_magic('store', 'recall_score_wt_BLDA2')


# ###### Bagging with Decision Trees

# In[59]:


feature_selection=VarianceThreshold()
clf = BaggingClassifier(DecisionTreeClassifier()) 

pipe_clf = make_pipeline(feature_selection,clf)
pipe_clf.fit(X_train_scaled, y_train)
y_pred_BDT2 = pipe_clf.predict(X_test_scaled)


# In[60]:


print('********Bagging Classifier with DT, Standard Scaled, Variance Threshold','********')

f1_score_micro_BDT2 = metrics.f1_score(y_test, y_pred_BDT2, average='micro')
f1_score_wt_BDT2 = metrics.f1_score(y_test,y_pred_BDT2,average='weighted')
precision_score_wt_BDT2 = metrics.precision_score(y_test, y_pred_BDT2, average='weighted')
recall_score_wt_BDT2 = metrics.recall_score(y_test, y_pred_BDT2, average='weighted')

print('F1-score_micro = ',f1_score_micro_BDT2)
print('F1-score = ',f1_score_wt_BDT2)
print('Precision = ',precision_score_wt_BDT2)
print('Recall Score = ',recall_score_wt_BDT2)

get_ipython().run_line_magic('store', 'f1_score_micro_BDT2')
get_ipython().run_line_magic('store', 'f1_score_wt_BDT2')
get_ipython().run_line_magic('store', 'precision_score_wt_BDT2')
get_ipython().run_line_magic('store', 'recall_score_wt_BDT2')


# In[61]:


import pickle
algorithm_3 = {'f1_score_micro_AB1':f1_score_micro_AB1,
'f1_score_wt_AB1': f1_score_wt_AB1,
'precision_score_wt_AB1': precision_score_wt_AB1,
'recall_score_wt_AB1':recall_score_wt_AB1,
'f1_score_micro_AB2':f1_score_micro_AB2,
'f1_score_wt_AB2':f1_score_wt_AB2,
'precision_score_wt_AB2':precision_score_wt_AB2,
'recall_score_wt_AB2':recall_score_wt_AB2,
'f1_score_micro_AB3':f1_score_micro_AB3,
'f1_score_wt_AB3':f1_score_wt_AB3,
'precision_score_wt_AB3':precision_score_wt_AB3,
'recall_score_wt_AB3':recall_score_wt_AB3,
'f1_score_micro_MLP1':f1_score_micro_MLP1,
'f1_score_wt_MLP1':f1_score_wt_MLP1,
'precision_score_wt_MLP1':precision_score_wt_MLP1,
'recall_score_wt_MLP1':recall_score_wt_MLP1,
'f1_score_micro_MLP2':f1_score_micro_MLP2,
'f1_score_wt_MLP2':f1_score_wt_MLP2,
'precision_score_wt_MLP2':precision_score_wt_MLP2,
'recall_score_wt_MLP2':recall_score_wt_MLP2,
'f1_score_micro_MLP3':f1_score_micro_MLP3,
'f1_score_wt_MLP3':f1_score_wt_MLP3,
'precision_score_wt_MLP3':precision_score_wt_MLP3,
'recall_score_wt_MLP3':recall_score_wt_MLP3,
'f1_score_micro_BGKNN1':f1_score_micro_BGKNN1,
'f1_score_wt_BGKNN1':f1_score_wt_BGKNN1,
'precision_score_wt_BGKNN1':precision_score_wt_BGKNN1,
'recall_score_wt_BGKNN1':recall_score_wt_BGKNN1,
'f1_score_micro_BGET2':f1_score_micro_BGET2,
'f1_score_wt_BGET2':f1_score_wt_BGET2,
'precision_score_wt_BGET2':precision_score_wt_BGET2,
'recall_score_wt_BGET2':recall_score_wt_BGET2,
'f1_score_micro_BRF2':f1_score_micro_BRF2,
'f1_score_wt_BRF2':f1_score_wt_BRF2,
'precision_score_wt_BGRF2':precision_score_wt_BGRF2,
'recall_score_wt_BGRF2':recall_score_wt_BGRF2,
'f1_score_micro_BLDA2':f1_score_micro_BLDA2,
'f1_score_wt_BLDA2':f1_score_wt_BLDA2,
'precision_score_wt_BLDA2':precision_score_wt_BLDA2,
'recall_score_wt_BLDA2':recall_score_wt_BLDA2,
'f1_score_micro_BDT2':f1_score_micro_BDT2,
'f1_score_wt_BDT2':f1_score_wt_BDT2,
'precision_score_wt_BDT2':precision_score_wt_BDT2,
'recall_score_wt_BDT2':recall_score_wt_BDT2
}

pickle.dump(algorithm_3,open("algorithm_3.p","wb"))

