# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:20:43 2021

@author: zsqmo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#***************************** Please choose the dataset **************************
# Load the data
data = pd.read_csv('endterm1.csv')
#data = pd.read_csv('midterm1.csv')

# build a new variable called 'deal_status_bi' and convert 'won' to 1, 'lost' and 'open' to 0
data['deal_status_bi'] = data['DEAL_STATUS']
data.loc[data['DEAL_STATUS']=='won','deal_status_bi']=1
data.loc[data['DEAL_STATUS']=='lost','deal_status_bi']=0
data.loc[data['DEAL_STATUS']=='open','deal_status_bi']=0

# convert column names from upper case to lower case
data.columns = data.columns.str.lower()

# set important variables and target
variables = ['activities_completed', 'calls_made', 'emails_sent',
       'meetings_conducted', 'tasks_completed', 'active_teachers',
       'tasks_created', 'tasks_created_per_teacher', 'problems_per_student',
       'proficient_per_student', 'mastered_per_student']

data['deal_status_bi'].value_counts()


#%%================================== EDA on original data  =======================================
# histograms
for i in variables:
    plt.figure(figsize =(10, 6))
    sns.distplot(data.loc[:,i], hist_kws={'alpha': 0.4, 'edgecolor':'blue'})
    plt.figtext(.5,.9,f'{i} Histogram', fontsize=20, ha='center')
    plt.show()

# boxplot
for i in variables:
    rows = data['deal_status_bi']<=1
    sns.boxplot(x = data.loc[rows, 'deal_status_bi'], y = data.loc[rows, i])
    plt.figtext(.5,.9,f'{i}', fontsize=18, ha='center')
    sns.despine()
    plt.show()
    

#%% ================================= EDA on transformed data ====================================

from sklearn.preprocessing import PowerTransformer
yj_transf_plot = PowerTransformer(method='yeo-johnson').fit(data[variables])
transf_plot = yj_transf_plot.transform(data[variables])
transf_df = pd.DataFrame(transf_plot, columns = ['activities_completed_yj', 'calls_made_yj', 'emails_sent_yj',
       'meetings_conducted_yj', 'tasks_completed_yj', 'active_teachers_yj',
       'tasks_created_yj', 'tasks_created_per_teacher_yj', 'problems_per_student_yj',
       'proficient_per_student_yj', 'mastered_per_student_yj'])

# transformed histograms
for i in transf_df:
    plt.figure(figsize =(10, 6))
    sns.distplot(transf_df.loc[:,i], hist_kws={'alpha': 0.4, 'edgecolor':'blue'})
    plt.figtext(.5,.9,f'{i} Histogram', fontsize=20, ha='center')
    plt.show()

# transformed boxplot
for i in transf_df:
    rows = data['deal_status_bi']<=1
    sns.boxplot(x = data.loc[rows, 'deal_status_bi'], y = transf_df.loc[rows, i])
    plt.figtext(.5,.9,f'{i}', fontsize=18, ha='center')
    sns.despine()
    plt.show()
    
#%% ================================= logistic regression on original data ========================
import statsmodels.api as sm
y_value = sm.add_constant(data[variables])
logit_model=sm.Logit(data['deal_status_bi'].astype(int),y_value)
result=logit_model.fit()
print(result.summary())


#%% ================================= Logistic regression on transformed data =========================
y_value_yj = sm.add_constant(transf_df)
logit_model_yj=sm.Logit(data['deal_status_bi'].astype(int),y_value_yj)
result_yj=logit_model_yj.fit()
print(result_yj.summary())

# prediction
# n,m = transf_df.shape
# transf_pre = np.hstack((np.ones((n,1)), transf_df))
# y_pred = result_yj.predict(transf_pre)


#%%=============================== Logistic regression (two variables) =================================
variables_sorted = ['active_teachers', 'proficient_per_student']
variables_sorted_yj = ['active_teachers_yj', 'proficient_per_student_yj']

y_value_sorted = sm.add_constant(data[variables_sorted])
logit_model_sorted=sm.Logit(data['deal_status_bi'].astype(int),y_value_sorted)
result_sorted=logit_model_sorted.fit()
print(result_sorted.summary())

y_value_sorted_yj = sm.add_constant(transf_df[variables_sorted_yj])
logit_model_sorted_yj=sm.Logit(data['deal_status_bi'].astype(int),y_value_sorted_yj)
result_sorted_yj=logit_model_sorted_yj.fit()
print(result_sorted_yj.summary())

n,m = data[variables_sorted].shape
variables_sorted_constant = np.hstack((np.ones((n,1)), data[variables_sorted]))
y_pred_sorted = result_sorted.predict(variables_sorted_constant)


#%% Combinations of "active_teacher" and "proficient_per_student"
import math
prob = 0.8
odds = math.log(prob/(1-prob))
active_teachers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
proficient_per_student = []
# end-term equation: odds = -2.3371+0.6122*active_teachers+0.2049*proficient_per_student = 1.38629
# midterm  equation: odds = -1.7444+0.6840*active_teachers+0.1988*proficient_per_student = 1.38629

for i in active_teachers:
    proficient_per_student_result = (odds - (result_sorted.params[0]) - result_sorted.params[1]*i)/result_sorted.params[2]
    proficient_per_student.append(proficient_per_student_result)





