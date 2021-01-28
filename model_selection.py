# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:03:48 2021

@author: zsqmo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
sns.set()

#***************************** Please choose the dataset **************************
# Load the data
#data = pd.read_csv('endterm1.csv')
data = pd.read_csv('midterm1.csv')

# build a new variable called 'deal_status_bi' and convert 'won' to 1, 'lost' and 'open' to 0
data['deal_status_bi'] = data['DEAL_STATUS']
data.loc[data['DEAL_STATUS']=='won','deal_status_bi']=1
data.loc[data['DEAL_STATUS']=='lost','deal_status_bi']=0
data.loc[data['DEAL_STATUS']=='open','deal_status_bi']=0
data['deal_status_bi'] = data['deal_status_bi'].astype('int')

# convert column names from upper case to lower case
data.columns = data.columns.str.lower()

# set important variables and target
variables = ['activities_completed', 'calls_made', 'emails_sent',
       'meetings_conducted', 'tasks_completed', 'active_teachers',
       'tasks_created', 'tasks_created_per_teacher', 'problems_per_student',
       'proficient_per_student', 'mastered_per_student']
target = 'deal_status_bi'


#%% split the dataset into training data and test data using original dataset
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test=train_test_split(data[variables], data[target], train_size=0.8,random_state=0)


#%% =========================================================================================================
# ===================== logistic regression on original dataset (with intercept)===========================
# ==========================================================================================================
#%% fit and predict 
import statsmodels.api as sm
X_train_new = sm.add_constant(X_train)
logit_model=sm.Logit(y_train, X_train_new)
result=logit_model.fit()

X_test_new = sm.add_constant(X_test)
y_probs = result.predict(X_test_new)
y_preds_1 = y_probs.replace(y_probs.where(y_probs>=0.5),1)
y_preds = y_preds_1.replace(y_preds_1.where(y_preds_1<0.5),0)
print(result.summary())

# show the accuracy report
print(classification_report(y_test, y_preds, digits=3))
print(confusion_matrix(y_test, y_preds))

# feature importance plot to show the important variables
def plot_coefficients(model, labels):
    coef = model.params.values
    table = pd.Series(coef.ravel(), index = labels).sort_values(ascending=True, inplace=False)
    
    all_ = True
    if len(table) > 20:
        reference = pd.Series(np.abs(coef.ravel()), index = labels).sort_values(ascending=False, inplace=False)
        reference = reference.iloc[:20]
        table = table[reference.index]
        table = table.sort_values(ascending=True, inplace=False)
        all_ = False
        

    fig, ax = fig, ax = plt.subplots()
    table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    if all_:
        ax.set_title('Estimated coefficients', fontsize=14)
    else: 
        ax.set_title('Estimated coefficients (20 largest in absolute value)', fontsize=14)
    sns.despine()
    return fig, ax

plot_coefficients(result, X_train_new.columns)



#%% model selection 
# using gridsearch to search for the best parameters
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01,0.1,0.5, 1, 5, 10, 20, 50, 100],
    'penalty': ['l1','l2', 'elasticnet', None],
    'fit_intercept': [False, True],
    'class_weight': ['balanced', None]
}

from sklearn.linear_model import LogisticRegression
log_reg_cv = LogisticRegression(solver='liblinear')
grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
grid_search.fit(X_train, y_train)
model_opt = grid_search.best_estimator_
print(model_opt)

# fit and predict
logit = model_opt.fit(X_train, y_train)
y_preds = logit.predict(X_test)

# show the accuracy report
print(classification_report(y_test, y_preds, digits=3))
print(confusion_matrix(y_test, y_preds))

# feature importance plot to show the important variables
def plot_coefficients2(model, labels):
    coef = model.coef_ 
    table = pd.Series(coef.ravel(), index = labels).sort_values(ascending=True, inplace=False)
    
    all_ = True
    if len(table) > 20:
        reference = pd.Series(np.abs(coef.ravel()), index = labels).sort_values(ascending=False, inplace=False)
        reference = reference.iloc[:20]
        table = table[reference.index]
        table = table.sort_values(ascending=True, inplace=False)
        all_ = False
        

    fig, ax = fig, ax = plt.subplots()
    table.T.plot(kind='barh', edgecolor='black', width=0.7, linewidth=.8, alpha=0.9, ax=ax)
    ax.tick_params(axis=u'y', length=0) 
    if all_:
        ax.set_title('Estimated coefficients', fontsize=14)
    else: 
        ax.set_title('Estimated coefficients (20 largest in absolute value)', fontsize=14)
    sns.despine()
    return fig, ax

plot_coefficients2(logit, X_train.columns)

#%% average accuracy score
scores = []
for i in np.arange(30):
    X_train, X_test, y_train, y_test = train_test_split(data[variables], data[target], train_size=0.8)
    log_reg_cv = LogisticRegression(solver='liblinear')
    grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
    grid_search.fit(X_train, y_train)
    model_opt = grid_search.best_estimator_
    logit = model_opt.fit(X_train, y_train)
    y_preds = logit.predict(X_test)
    scores.append(accuracy_score(y_test, y_preds))
print(np.average(scores))
print(np.std(scores))





#%%======================================================================================================
# ====================Logistic regression (customised variables) =======================================
#=========================================================================================================
variables_sort = ['active_teachers', 'proficient_per_student']
#*********** Please choose the following alternatives to run these 4 cells ************************

#variables_sort = ['active_teachers', 'proficient_per_student', 'problems_per_student']
#variables_sort = ['active_teachers', 'proficient_per_student', 'mastered_per_student']
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(data[variables_sort], data[target], train_size=0.8,random_state=0)

#%% fit and predict 
import statsmodels.api as sm
X_train_2_new = sm.add_constant(X_train_2)
logit_model_2=sm.Logit(y_train_2, X_train_2_new)
result_2=logit_model_2.fit()

X_test_2_new = sm.add_constant(X_test_2)
y_probs = result_2.predict(X_test_2_new)
y_preds_1 = y_probs.replace(y_probs.where(y_probs>=0.5),1)
y_preds = y_preds_1.replace(y_preds_1.where(y_preds_1<0.5),0)
print(result_2.summary())

# show the accuracy report
print(classification_report(y_test_2, y_preds, digits=3))
print(confusion_matrix(y_test_2, y_preds))

#%% model selection 
# using gridsearch to search for the best parameters
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01,0.1,0.5, 1, 5, 10, 20, 50, 100],
    'penalty': ['l1','l2', 'elasticnet', None],
    'fit_intercept': [False, True],
    'class_weight': ['balanced', None]
}

from sklearn.linear_model import LogisticRegression
log_reg_cv = LogisticRegression(solver='liblinear')
grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
grid_search.fit(X_train_2, y_train_2)
model_opt = grid_search.best_estimator_
print(model_opt)

# fit and predict
logit = model_opt.fit(X_train_2, y_train_2)
y_preds = logit.predict(X_test_2)

# show the accuracy report
print(classification_report(y_test_2, y_preds, digits=3))
print(confusion_matrix(y_test_2, y_preds))

#%% average accuracy score
scores = []
for i in np.arange(30):
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(data[variables_sort], data[target], train_size=0.8)
    log_reg_cv = LogisticRegression(solver='liblinear')
    grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
    grid_search.fit(X_train_2, y_train_2)
    model_opt = grid_search.best_estimator_
    logit = model_opt.fit(X_train_2, y_train_2)
    #print(logit)
    y_preds = logit.predict(X_test_2)
    scores.append(accuracy_score(y_test_2, y_preds))
print(np.average(scores))
print(np.std(scores))






#%% ===========================================================================================================
# ===================== logistic regression on original dataset (without intercept)===========================
#==================================================================================================================
#%% fit and predict 
logit_model_ni=sm.Logit(y_train, X_train)
result_ni=logit_model_ni.fit()
print(result_ni.summary())

y_probs = result_ni.predict(X_test)
y_preds_1 = y_probs.replace(y_probs.where(y_probs>=0.5),1)
y_preds = y_preds_1.replace(y_preds_1.where(y_preds_1<0.5),0)

# show the accuracy report
print(classification_report(y_test, y_preds, digits=3))
print(confusion_matrix(y_test, y_preds))
plot_coefficients(result_ni, X_train.columns)

#%%  model selection 
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01,0.1,0.5, 1, 5, 10, 20, 50, 100],
    #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1','l2', 'elasticnet', None],
    'class_weight': ['balanced', None]
}

from sklearn.linear_model import LogisticRegression
log_reg_cv = LogisticRegression(solver = 'liblinear', fit_intercept = False)
grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
grid_search.fit(X_train, y_train)
model_opt = grid_search.best_estimator_
print(model_opt)

# fit and predict
logit = model_opt.fit(X_train, y_train)
y_preds = logit.predict(X_test)

# show the accuracy report
print(classification_report(y_test, y_preds, digits=3))
print(confusion_matrix(y_test, y_preds))
plot_coefficients2(logit, X_train.columns)

#%% average accuracy score
scores = []
for i in np.arange(100):
    X_train, X_test, y_train, y_test = train_test_split(data[variables], data[target], train_size=0.8)
    log_reg_cv = LogisticRegression(solver = 'liblinear', fit_intercept = False)
    grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
    grid_search.fit(X_train, y_train)
    model_opt = grid_search.best_estimator_
    logit = model_opt.fit(X_train, y_train)
    y_preds = logit.predict(X_test)
    scores.append(accuracy_score(y_test, y_preds))
print(np.average(scores))






#%% ===============================================================================================
#======================= Logistic regression on transformed dataset ================================
#===================================================================================================
# using yeo-johnson transformation for normality purpose
from sklearn.preprocessing import PowerTransformer
yj_transf = PowerTransformer(method='yeo-johnson').fit(data.loc[:, variables])
variables_yj = yj_transf.transform(data[variables])
variables_yj2 = pd.DataFrame(variables_yj, columns = ['activities_completed_yj', 'calls_made_yj', 'emails_sent_yj',
       'meetings_conducted_yj', 'tasks_completed_yj', 'active_teachers_yj',
       'tasks_created_yj', 'tasks_created_per_teacher_yj', 'problems_per_student_yj',
       'proficient_per_student_yj', 'mastered_per_student_yj'])

#%% split the dataset into training data and test data using transformed dataset
X_train_yj,X_test_yj, y_train_yj, y_test_yj=train_test_split(variables_yj2, data[target], train_size=0.8,random_state=0)


#%% fit and predict
import statsmodels.api as sm
X_train_yj_new = sm.add_constant(X_train_yj)
logit_model=sm.Logit(y_train_yj, X_train_yj_new)
result=logit_model.fit()

X_test_yj_new = sm.add_constant(X_test_yj)
y_probs = result.predict(X_test_yj_new)
y_preds_1 = y_probs.replace(y_probs.where(y_probs>=0.5),1)
y_preds = y_preds_1.replace(y_preds_1.where(y_preds_1<0.5),0)

# show the accuracy report
print(classification_report(y_test_yj, y_preds, digits=3))
print(confusion_matrix(y_test_yj, y_preds))

# feature importance plot to show the important variables
plot_coefficients(result, X_train_yj_new.columns)

#%% model selection 
# using gridsearch to search for the best parameters
from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.01,0.1,0.5, 1, 5, 10, 20, 50, 100],
    'penalty': ['l1','l2', 'elasticnet', None],
    'fit_intercept': [False, True],
    'class_weight': ['balanced', None]
}

from sklearn.linear_model import LogisticRegression
log_reg_cv = LogisticRegression(solver='liblinear')
grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
grid_search.fit(X_train_yj, y_train_yj)
model_opt = grid_search.best_estimator_
print(model_opt)

# fit and predict
logit = model_opt.fit(X_train_yj, y_train_yj)
y_preds = logit.predict(X_test_yj)

# show the accuracy report
print(classification_report(y_test_yj, y_preds, digits=3))
print(confusion_matrix(y_test_yj, y_preds))

# feature importance plot to show the important variables
plot_coefficients2(logit, X_train_yj.columns)

#%% average accuracy score
scores = []
for i in np.arange(30):
    X_train_yj, X_test_yj, y_train_yj, y_test_yj = train_test_split(variables_yj2, data[target], train_size=0.8)
    log_reg_cv = LogisticRegression(solver = 'liblinear')
    grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
    grid_search.fit(X_train_yj, y_train_yj)
    model_opt = grid_search.best_estimator_
    logit = model_opt.fit(X_train_yj, y_train_yj)
    y_preds = logit.predict(X_test_yj)
    scores.append(accuracy_score(y_test_yj, y_preds))
print(np.average(scores))
print(np.std(scores))







#%% ===============================================================================================
#======================= Logistic regression on transformed dataset (customised variables) ================================
#===================================================================================================
variables_sort_yj = ['active_teachers_yj', 'proficient_per_student_yj']

#*********** Please choose the following alternatives to run this cell ************************
#variables_sort = ['active_teachers_yj', 'proficient_per_student_yj', 'problems_per_student_yj']
#variables_sort = ['active_teachers_yj', 'proficient_per_student_yj', 'mastered_per_student_yj']
scores = []
for i in np.arange(30):
    X_train_2_yj, X_test_2_yj, y_train_2_yj, y_test_2_yj = train_test_split(variables_yj2[variables_sort_yj], data[target], train_size=0.8)
    log_reg_cv = LogisticRegression(solver = 'liblinear')
    grid_search = GridSearchCV(log_reg_cv, param_grid, return_train_score=True, cv = 10, n_jobs=8, scoring='neg_log_loss')
    grid_search.fit(X_train_2_yj, y_train_2_yj)
    model_opt = grid_search.best_estimator_
    logit = model_opt.fit(X_train_2_yj, y_train_2_yj)
    y_preds = logit.predict(X_test_2_yj)
    scores.append(accuracy_score(y_test_2_yj, y_preds))
print(np.average(scores))
print(np.std(scores))












