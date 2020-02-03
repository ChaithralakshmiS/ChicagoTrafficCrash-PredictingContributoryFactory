
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pickle


# In[3]:


algorithm_1 = pickle.load(open("algorithm_1.p","rb"))
algorithm_2 = pickle.load(open("algorithm_2.p","rb"))
algorithm_3 = pickle.load(open("algorithm_3.p","rb"))


# In[4]:


sampled_variables = pickle.load(open("sampled_variables.p", 'rb'))


# In[5]:


for key in algorithm_1.keys():
    var = key
    exec(var + " = algorithm_1[key]")


# In[6]:


for key in algorithm_2.keys():
    var = key
    exec(var + " = algorithm_2[key]")


# In[7]:


for key in algorithm_3.keys():
    var = key
    exec(var + " = algorithm_3[key]")


# In[8]:


for key in sampled_variables.keys():
    var = key
    exec(var + " = sampled_variables[key]")


# In[9]:


f1_score_wt_pca = [f1_score_wt_DT1, f1_score_wt_ET1, f1_score_wt_LDA1,
                  f1_score_wt_AB1, f1_score_wt_MLP1, f1_score_wt_BGKNN1,
                  f1_score_wt_1_GNB1, f1_score_wt_1_LR1, f1_score_wt_1_RF1
                  ]
f1_score_micro_pca = [f1_score_micro_DT1, f1_score_micro_ET1, f1_score_micro_LDA1,
                      f1_score_micro_AB1, f1_score_micro_MLP1, f1_score_micro_BGKNN1,
                      f1_score_micro_1_GNB1, f1_score_micro_1_LR1, f1_score_micro_1_RF1, 
                      
                     ]
precision_score_pca = [precision_score_wt_DT1, precision_score_wt_ET1, precision_score_wt_LDA1,
                      precision_score_wt_AB1, precision_score_wt_MLP1, precision_score_wt_BGKNN1,
                      precision_score_wt_1_GNB1, precision_score_wt_1_LR1, precision_score_wt_1_RF1
                      ]
recall_score_pca = [recall_score_wt_DT1, recall_score_wt_ET1, recall_score_wt_LDA1,   
                   recall_score_wt_AB1, recall_score_wt_MLP1, recall_score_wt_BGKNN1,
                   recall_score_wt_1_GNB1,recall_score_wt_1_LR1, recall_score_wt_1_RF1
                    ]


# In[10]:


f1_score_wt_vt = [f1_score_wt_DT2, f1_score_wt_ET2, f1_score_wt_LDA2,
                  f1_score_wt_AB2, f1_score_wt_MLP2,
                  f1_score_wt_BGET2, f1_score_wt_BRF2,f1_score_wt_BLDA2, f1_score_wt_BDT2,
                  f1_score_wt_2_GNB2, f1_score_wt_2_LR2, f1_score_wt_2_RF2
                 ]
f1_score_micro_vt = [f1_score_micro_DT2, f1_score_micro_ET2, f1_score_micro_LDA2,
                    f1_score_micro_AB2, f1_score_micro_MLP2,
                    f1_score_micro_BGET2, f1_score_micro_BRF2, f1_score_micro_BLDA2, f1_score_micro_BDT2,
                     f1_score_micro_2_GNB2, f1_score_micro_2_LR2, f1_score_micro_2_RF2
                    ]
precision_score_vt = [precision_score_wt_DT2, precision_score_wt_ET2, precision_score_wt_LDA2,
                     precision_score_wt_AB1, precision_score_wt_MLP1,
                     precision_score_wt_BGET2,precision_score_wt_BGRF2, precision_score_wt_BLDA2, precision_score_wt_BDT2,
                      precision_score_wt_2_GNB2, precision_score_wt_2_LR2 , precision_score_wt_2_RF2
                     ]
recall_score_vt = [recall_score_wt_DT2, recall_score_wt_ET2, recall_score_wt_LDA2,
                  recall_score_wt_AB2, recall_score_wt_MLP2,
                   recall_score_wt_BGET2,recall_score_wt_BGRF2, recall_score_wt_BLDA2, recall_score_wt_BDT2,
                   recall_score_wt_2_GNB2, recall_score_wt_2_LR2, recall_score_wt_2_RF2
                  ]


# In[9]:


f1_score_wt_svd = [f1_score_wt_DT3, f1_score_wt_ET3, f1_score_wt_LDA3,
                   f1_score_wt_AB3, f1_score_wt_MLP3,
                   f1_score_wt_2_GNB3, f1_score_tsvd_wt_LR3, f1_score_wt_2_RF3
                  ]
f1_score_micro_svd = [f1_score_micro_DT3, f1_score_micro_ET3, f1_score_micro_LDA3,
                     f1_score_micro_AB3, f1_score_micro_MLP3,
                      f1_score_micro_2_GNB3, f1_score_tsvd_micro_LR3, f1_score_micro_2_RF3
                     
                     ]
precision_score_svd = [precision_score_wt_DT3, precision_score_wt_ET3, precision_score_wt_LDA3,
                      precision_score_wt_AB3, precision_score_wt_MLP3,
                       precision_score_wt_2_GNB3, precision_score_tsvd_wt_LR3, precision_score_wt_2_RF3
                      ]
recall_score_svd = [recall_score_wt_DT3, recall_score_wt_ET3, recall_score_wt_LDA3,
                   recall_score_wt_AB3, recall_score_wt_MLP3,
                    recall_score_wt_2_GNB3, recall_score_tsvd_wt_LR3, recall_score_wt_2_RF3
                   ]


# ### Comparison of Accuracy and F1Scores

# In[22]:


import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(250, 60)

f1_score_wt_vt = [f1_score_wt_DT2, f1_score_wt_ET2, f1_score_wt_LDA2,
                  f1_score_wt_AB2, f1_score_wt_MLP2,
                  f1_score_wt_BGET2, f1_score_wt_BRF2,f1_score_wt_BLDA2, f1_score_wt_BDT2,
                  f1_score_wt_2_GNB2, f1_score_wt_2_LR2, f1_score_wt_2_RF2
                 ]

f1_score_micro_vt = [f1_score_micro_DT2, f1_score_micro_ET2, f1_score_micro_LDA2,
                    f1_score_micro_AB2, f1_score_micro_MLP2,
                    f1_score_micro_BGET2, f1_score_micro_BRF2, f1_score_micro_BLDA2, f1_score_micro_BDT2,
                     f1_score_micro_2_GNB2, f1_score_micro_2_LR2, f1_score_micro_2_RF2
                    ]

ind = np.arange(len(f1_score_wt_vt))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(ind + 0.00, f1_score_wt_vt, width, color='red', label='F1-Weighted')
rects2 = ax.bar(ind + 0.25, f1_score_micro_vt, width, color='darkblue', label='Accuracy')


ax.set_title('F1 and Accuracy Scores Comparison', fontsize=14)
ax.set_xlabel("Algorithms", fontsize = 12)
ax.set_ylabel("F1 and Accuracy", fontsize = 12)

ax.set_xticks(ind)
ax.set_xticklabels(('DT','ET', 'LDA','AB', 'MLP', 'BGET', 'BRF', 'BLDA', 'BDT', 'GNB', 'LR', 'RF'), fontsize=8)
ax.legend(loc='lower right')


# ### Comparison between F1 scores obatained between different dimensionality reduction techniques - PCA, VarianceThreshold, TruncatedSVD 
# 
# 

# In[13]:


import matplotlib.pyplot as plt


f1_score_wt_pca_upd = [f1_score_wt_DT1, f1_score_wt_ET1, f1_score_wt_LDA1,
                  f1_score_wt_AB1, f1_score_wt_MLP1,
                  f1_score_wt_1_GNB1, f1_score_wt_1_LR1, f1_score_wt_1_RF1]

f1_score_wt_vt_upd = [f1_score_wt_DT2, f1_score_wt_ET2, f1_score_wt_LDA2,
                  f1_score_wt_AB2, f1_score_wt_MLP2,
                  f1_score_wt_2_GNB2, f1_score_wt_2_LR2, f1_score_wt_2_RF2]

f1_score_wt_svd_upd = [f1_score_wt_DT3, f1_score_wt_ET3, f1_score_wt_LDA3,
                   f1_score_wt_AB3, f1_score_wt_MLP3,
                   f1_score_wt_2_GNB3, f1_score_tsvd_wt_LR3, f1_score_wt_2_RF3]


axes = plt.gca()
axes.set_ylim([0,0.6])

x=np.arange(1,9)
# x = np.linspace(0.1, 2 * np.pi, 8)
plt.yticks(np.arange(0, 0.7, 0.05))

markerline1, stemlines, _ = plt.stem(x, f1_score_wt_pca_upd, '-.')
plt.setp(markerline1, 'markerfacecolor', 'b', label='PCA')

markerline2, stemlines, _ = plt.stem(x, f1_score_wt_vt_upd, '-.')
plt.setp(markerline2, 'markerfacecolor', 'r', label='VT')

markerline3, stemlines, _ = plt.stem(x, f1_score_wt_svd_upd, '-.')
plt.setp(markerline3, 'markerfacecolor', 'g', label='SVD')
plt.legend(loc='lower right')


axes.set_xticklabels([' ', 'DT','ET', 'LDA','AB', 'MLP', 'GNB', 'LR', 'RF'])
axes.set_title("F1 scores - PCA, VT, SVD", fontsize=14)
plt.xlabel("Algorithms", fontsize = 12)
plt.ylabel("F1 Scores", fontsize = 12)

plt.show()


# ###### Comparison of F1 Scores with and without Sampling

# In[21]:


f1_score_with_sampling = [f1_score_wt_DT_ds, f1_score_wt_ET_ds, f1_score_wt_LDA_ds,
                          f1_score_wt_AB_ds, f1_score_wt_MLP_ds,
                          f1_score_wt_BET_ds, f1_score_wt_BRF_ds, f1_score_wt_BLDA_ds, f1_score_wt_BDT_ds,
                          f1_score_wt_NB_ds, f1_score_wt_LR_ds, f1_score_wt_RF_ds
                         ]

f1_score_without_sampling = [f1_score_micro_DT2, f1_score_micro_ET2, f1_score_micro_LDA2,
                    f1_score_micro_AB2, f1_score_micro_MLP2,
                    f1_score_micro_BGET2, f1_score_micro_BRF2, f1_score_micro_BLDA2, f1_score_micro_BDT2,
                     f1_score_micro_2_GNB2, f1_score_micro_2_LR2, f1_score_micro_2_RF2]

ind = np.arange(len(f1_score_with_sampling))
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(ind + 0.00, f1_score_with_sampling, width, color='pink', label='F1-W-Sampling')
rects2 = ax.bar(ind + 0.25, f1_score_without_sampling, width, color='seagreen', label='F1-WO-Sampling')


ax.set_title('F1 Scores Comparison - with and without Sampling', fontsize=14)
ax.set_xlabel("Algorithms", fontsize = 12)
ax.set_ylabel("F1 Scores", fontsize = 12)

ax.set_xticks(ind)
ax.set_xticklabels(('DT','ET', 'LDA','AB', 'MLP', 'BGET', 'BRF', 'BLDA', 'BDT', 'GNB', 'LR', 'RF'), fontsize=8)
ax.legend(loc='lower right')


# ### References
# #### https://stackoverflow.com/questions/41538681/plot-two-lists-with-different-color-with-stem
