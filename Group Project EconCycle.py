# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 00:21:33 2020

@author: Lu Yuan
"""
import pandas as pd 
data = pd.read_csv('MLF_GP2_EconCycle.csv')
data.head()

#discard the Date column
data = data.iloc[:,1:]
data.head()

#additional features
data['CP1M_T2Y']= data['CP1M']/data['T2Y Index']
data['CP3M_T2Y']= data['CP3M']/data['T2Y Index']
data['CP6M_T2Y']= data['CP6M']/data['T2Y Index']

data['CP1M_T3Y']= data['CP1M']/data['T3Y Index']
data['CP3M_T3Y']= data['CP3M']/data['T3Y Index']
data['CP6M_T3Y']= data['CP6M']/data['T3Y Index']

data['CP1M_T5Y']= data['CP1M']/data['T5Y Index']
data['CP3M_T5Y']= data['CP3M']/data['T5Y Index']
data['CP6M_T5Y']= data['CP6M']/data['T5Y Index']

data['CP1M_T7Y']= data['CP1M']/data['T7Y Index']
data['CP3M_T7Y']= data['CP3M']/data['T7Y Index']
data['CP6M_T7Y']= data['CP6M']/data['T7Y Index']

data['CP1M_T10Y']= data['CP1M']/data['T10Y Index']
data['CP3M_T10Y']= data['CP3M']/data['T10Y Index']
data['CP6M_T10Y']= data['CP6M']/data['T10Y Index']

data.head()

# Part 1: Introduction/Exploratory Data Analysis

import matplotlib.pyplot as plt
#time series
data[['PCT 3MO FWD', 'PCT 6MO FWD', 'PCT 9MO FWD']].plot()
plt.title('Trend of the PCTs')
plt.ylabel('PCT')

#Scatterplot matrix
import seaborn as sns

data1_cols = ['T1Y Index','T2Y Index','T3Y Index','T5Y Index','T7Y Index','T10Y Index',
              'CP1M','CP3M','CP6M','CP1M_T1Y','CP3M_T1Y','CP6M_T1Y','PCT 3MO FWD',
              'PCT 6MO FWD','PCT 9MO FWD']
sns.pairplot(data[data1_cols])

#Heatmap
import numpy as np
cm1 = np.corrcoef(data[data1_cols].values.T)
sns.heatmap(cm1, annot = True, fmt='.2f',annot_kws = {"size": 7},yticklabels=data1_cols, 
            xticklabels=data1_cols)
plt.tight_layout()
plt.title("Heatmap")
plt.show()

#Box plot
target = data[['PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD']]
sns.boxplot(data = target)
plt.xlabel('Attribute')
plt.ylabel('Quantile Ranges')
plt.title("Box plot")
plt.xticks(rotation = 90)
plt.show()

# Part 2: Preprocessing, feature extraction, feature selection

X = data[['T1Y Index','T2Y Index','T3Y Index','T5Y Index','T7Y Index','T10Y Index',
         'CP1M','CP3M','CP6M','CP1M_T1Y','CP3M_T1Y','CP6M_T1Y','CP1M_T2Y','CP3M_T2Y',
         'CP6M_T2Y','CP1M_T3Y','CP3M_T3Y','CP6M_T3Y','CP1M_T5Y','CP3M_T5Y','CP6M_T5Y',
         'CP1M_T7Y','CP3M_T7Y','CP6M_T7Y','CP1M_T10Y','CP3M_T10Y','CP6M_T10Y']]
y1 = data['PCT 3MO FWD']
y2 = data['PCT 6MO FWD']
y3 = data['PCT 9MO FWD']
y = data[['PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)  #???

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

y1_train= y_train.iloc[:,0]
y2_train=y_train.iloc[:,1]
y3_train=y_train.iloc[:,2]

y1_test=y_test.iloc[:,0]
y2_test=y_test.iloc[:,1]
y3_test=y_test.iloc[:,2]

Xcols=['T1Y Index','T2Y Index','T3Y Index','T5Y Index','T7Y Index','T10Y Index','CP1M','CP3M','CP6M','CP1M_T1Y','CP3M_T1Y','CP6M_T1Y',
       'CP1M_T2Y','CP3M_T2Y','CP6M_T2Y',
      'CP1M_T3Y','CP3M_T3Y','CP6M_T3Y',
      'CP1M_T5Y','CP3M_T5Y','CP6M_T5Y',
      'CP1M_T7Y','CP3M_T7Y','CP6M_T7Y',
      'CP1M_T10Y','CP3M_T10Y','CP6M_T10Y']
# ycols = ['PCT 3MO FWD','PCT 6MO FWD','PCT 9MO FWD']

##using PCA to do feature extraction and feature selection

from sklearn.decomposition import PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
#X_test_pca=pca.transform(X_test_std)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
print(np.cumsum(pca.explained_variance_ratio_))

pca=PCA(n_components = 4)
X_train_pca=pca.fit_transform(X_train_std)
X_test_pca=pca.transform(X_test_std)

# Part 3: Model fitting and evaluation & Part 4: Hyperparameter tuning

#LASSO Regression

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso

# Hyperparameter tuning for Y1 (PCT 3MO FWD) 

k_range = (0.0,0.2,0.4,0.6,0.8,1.0)
for k in k_range:
    lasso = Lasso(alpha=k)
    lasso.fit(X_train_std, y1_train)
    y1_pred_train = lasso.predict(X_train_std)
    y1_pred_test = lasso.predict(X_test_std)
    print('alpha=',k,' MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y1_train, y1_pred_train),
        mean_squared_error(y1_test, y1_pred_test)))
    print('alpha=',k ,' R^2 train: %.3f, test: %.3f' % (
        r2_score(y1_train, y1_pred_train),
        r2_score(y1_test, y1_pred_test)))
    
# Model fitting and evaluation for Y1 (PCT 3MO FWD) 

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
lasso = Lasso(alpha=0.0)
lasso.fit(X_train_std, y1_train)
y1_pred_train = lasso.predict(X_train_std)
y1_pred_test = lasso.predict(X_test_std)
lasso_coef = pd.DataFrame(lasso.coef_)
df_Xcols = pd.DataFrame(Xcols)
coefficient1 = pd.concat([df_Xcols, lasso_coef],axis=1, ignore_index=True)
coefficient1.columns = ['feature', 'coef']
print(coefficient1)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y1_train, y1_pred_train), 
                                       mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y1_train, y1_pred_train),
                                       r2_score(y1_test, y1_pred_test)))

# Hyperparameter tuning for Y2 (PCT 6MO FWD)

k_range = (0.0,0.2,0.4,0.6,0.8,1.0)
for k in k_range:
    lasso = Lasso(alpha=k)
    lasso.fit(X_train_std, y2_train)
    y2_pred_train = lasso.predict(X_train_std)
    y2_pred_test = lasso.predict(X_test_std)
    print('alpha=',k,'MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y2_train, y2_pred_train),
        mean_squared_error(y2_test, y2_pred_test)))
    print('alpha=',k ,'R^2 train: %.3f, test: %.3f' % (
        r2_score(y2_train, y2_pred_train),
        r2_score(y2_test, y2_pred_test)))
    
# Model fitting and evaluation for Y2 (PCT 6MO FWD)

lasso = Lasso(alpha=0.0)
lasso.fit(X_train_std, y2_train)
y2_pred_train = lasso.predict(X_train_std)
y2_pred_test = lasso.predict(X_test_std)
#print(lasso.coef_)
lasso_coef = pd.DataFrame(lasso.coef_)
df_Xcols = pd.DataFrame(Xcols)
coefficient2 = pd.concat([df_Xcols, lasso_coef],axis=1, ignore_index=True)
coefficient2.columns = ['feature', 'coef']
print(coefficient2)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y2_train, y2_pred_train), 
                                       mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y2_train, y2_pred_train),
                                       r2_score(y2_test, y2_pred_test)))

# Hyperparameter tuning for Y3 (PCT 9MO FWD)

k_range = (0.0,0.2,0.4,0.6,0.8,1.0)
for k in k_range:
    lasso = Lasso(alpha=k)
    lasso.fit(X_train_std, y3_train)
    y3_pred_train = lasso.predict(X_train_std)
    y3_pred_test = lasso.predict(X_test_std)
    print('alpha=',k,'MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y3_train, y3_pred_train),
        mean_squared_error(y3_test, y3_pred_test)))
    print('alpha=',k ,'R^2 train: %.3f, test: %.3f' % (
        r2_score(y3_train, y3_pred_train),
        r2_score(y3_test, y3_pred_test)))
    
# Model fitting and evaluation for Y3 (PCT 9MO FWD)

lasso = Lasso(alpha=0.0)
lasso.fit(X_train_std, y3_train)
y3_pred_train = lasso.predict(X_train_std)
y3_pred_test = lasso.predict(X_test_std)
#print(lasso.coef_)
lasso_coef = pd.DataFrame(lasso.coef_)
df_Xcols = pd.DataFrame(Xcols)
coefficient3 = pd.concat([df_Xcols, lasso_coef],axis=1, ignore_index=True)
coefficient3.columns = ['feature', 'coef']
print(coefficient3)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y3_train, y3_pred_train), 
                                       mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y3_train, y3_pred_train),
                                       r2_score(y3_test, y3_pred_test)))

# linear regression

from sklearn import linear_model
from sklearn.metrics import *
from math import *
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score

slr = linear_model.LinearRegression()

#Y1 PCT 3MO FWD

slr.fit(X_train_pca, y1_train)
y1_pred_train = slr.predict(X_train_pca)
y1_pred_test = slr.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y1_train, y1_pred_train),
        mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y1_train, y1_pred_train),
        r2_score(y1_test, y1_pred_test)))
print('Slope: ', slr.coef_)
print('Intercept: %.3f' % slr.intercept_)

#Y2 PCT 6MO FWD

slr.fit(X_train_pca, y2_train)
y2_pred_train = slr.predict(X_train_pca)
y2_pred_test = slr.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y2_train, y2_pred_train),
        mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y2_train, y2_pred_train),
        r2_score(y2_test, y2_pred_test)))
print('Slope: ', slr.coef_)
print('Intercept: %.3f' % slr.intercept_)

#Y3 PCT 9MO FWD
slr.fit(X_train_pca, y3_train)
y3_pred_train = slr.predict(X_train_pca)
y3_pred_test = slr.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y3_train, y3_pred_train),
        mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y3_train, y3_pred_train),
        r2_score(y3_test, y3_pred_test)))
print('Slope: ', slr.coef_)
print('Intercept: %.3f' % slr.intercept_)

#decision tree

#Y1 PCT 3MO FWD
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score

k_range = (1,2,3,5,10,15,20,25)
for k in k_range: 
    tree1 = DecisionTreeRegressor(max_depth = k, min_samples_leaf = 0.2, random_state = 42)
    tree1.fit(X_train_pca, y1_train)
    y1_pred_train = tree1.predict(X_train_pca)
    y1_pred_test = tree1.predict(X_test_pca)
    print('max_depth = ',k,' MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y1_train, y1_pred_train),
        mean_squared_error(y1_test, y1_pred_test)))
    print('max_depth = ',k ,' R^2 train: %.3f, test: %.3f' % (
        r2_score(y1_train, y1_pred_train),
        r2_score(y1_test, y1_pred_test)))
#     print('Accuracy train: %.3f, test: %.3f' % (
#         accuracy_score(y1_train, y1_pred_train), 
#         accuracy_score(y1_train, y1_pred_train)))

tree1 = DecisionTreeRegressor(max_depth = 3,min_samples_leaf = 0.2)
tree1.fit(X_train_pca, y1_train)

y1_pred_train = tree1.predict(X_train_pca)
y1_pred_test = tree1.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y1_train, y1_pred_train), 
                                       mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y1_train, y1_pred_train),
                                       r2_score(y1_test, y1_pred_test)))
# print('RMSE train: %.3f, test: %.3f' % (np.sqrt(mean_squared_error(y1_train, y1_pred_train)), 
#                                         np.sqrt(mean_squared_error(y1_test, y1_pred_test))))

#Y2 PCT 6MO FWD

k_range = (1,2,3,5,10,15,20,25)
for k in k_range: 
    tree2 = DecisionTreeRegressor(max_depth = k, min_samples_leaf = 0.2, random_state = 42)
    tree2.fit(X_train_pca, y2_train)
    y2_pred_train = tree2.predict(X_train_pca)
    y2_pred_test = tree2.predict(X_test_pca)
    print('max_depth = ',k,' MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y2_train, y2_pred_train),
        mean_squared_error(y2_test, y2_pred_test)))
    print('max_depth = ',k ,' R^2 train: %.3f, test: %.3f' % (
        r2_score(y2_train, y2_pred_train),
        r2_score(y2_test, y2_pred_test)))
    
tree2 = DecisionTreeRegressor(max_depth = 3,min_samples_leaf = 0.2)
tree2.fit(X_train_pca, y2_train)

y2_pred_train = tree2.predict(X_train_pca)
y2_pred_test = tree2.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y2_train, y2_pred_train), 
                                       mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y2_train, y2_pred_train),
                                       r2_score(y2_test, y2_pred_test)))

#Y3 PCT 9MO FWD

k_range = (1,2,3,5,10,15,20,25)
for k in k_range: 
    tree3 = DecisionTreeRegressor(max_depth = k, min_samples_leaf = 0.2, random_state = 42)
    tree3.fit(X_train_pca, y3_train)
    y3_pred_train = tree3.predict(X_train_pca)
    y3_pred_test = tree3.predict(X_test_pca)
    print('max_depth = ',k,' MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y3_train, y3_pred_train),
        mean_squared_error(y3_test, y3_pred_test)))
    print('max_depth = ',k ,' R^2 train: %.3f, test: %.3f' % (
        r2_score(y3_train, y3_pred_train),
        r2_score(y3_test, y3_pred_test)))
    
tree3 = DecisionTreeRegressor(max_depth = 3,min_samples_leaf = 0.2)
tree3.fit(X_train_pca, y3_train)

y3_pred_train = tree3.predict(X_train_pca)
y3_pred_test = tree3.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y3_train, y3_pred_train), 
                                       mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y3_train, y3_pred_train),
                                       r2_score(y3_test, y3_pred_test)))

# Part5: Ensembling

# Random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

 #Y1 PCT 3MO FWD  (PCA)

forest = RandomForestRegressor()
params_forest = {'n_estimators':[5,25,50,100,200],'max_features':['log2', 'auto', 'sqrt'],
             'min_samples_leaf':[1,2,5,50,100,200]}

grid_forest = GridSearchCV(estimator=forest,param_grid=params_forest,
                           scoring='neg_mean_squared_error',cv=10,verbose=1,n_jobs=-1)
grid_forest.fit(X_train_pca,y1_train)
best_model = grid_forest.best_estimator_
print(best_model)

y1_pred_train = best_model.predict(X_train_pca)
y1_pred_test = best_model.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y1_train, y1_pred_train), 
                                       mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y1_train, y1_pred_train),
                                       r2_score(y1_test, y1_pred_test)))

 #Y1 PCT 3MO FWD   (std)

forest = RandomForestRegressor()
params_forest = {'n_estimators':[5,25,50,100,200],'max_features':['log2', 'auto', 'sqrt'],
             'min_samples_leaf':[1,2,5,50,100,200]}

grid_forest = GridSearchCV(estimator=forest,param_grid=params_forest,
                           scoring='neg_mean_squared_error',cv=10,verbose=1,n_jobs=-1)
grid_forest.fit(X_train_std,y1_train)
best_model = grid_forest.best_estimator_
print(best_model)

y1_pred_train = best_model.predict(X_train_std)
y1_pred_test = best_model.predict(X_test_std)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y1_train, y1_pred_train), 
                                       mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y1_train, y1_pred_train),
                                       r2_score(y1_test, y1_pred_test)))

#Y2 PCT 6MO FWD  (PCA)

forest = RandomForestRegressor()
params_forest = {'n_estimators':[5,25,50,100,200],'max_features':['log2', 'auto', 'sqrt'],
             'min_samples_leaf':[1,2,5,50,100,200]}

grid_forest = GridSearchCV(estimator=forest,param_grid=params_forest,
                           scoring='neg_mean_squared_error',cv=10,verbose=1,n_jobs=-1)
grid_forest.fit(X_train_pca,y2_train)
best_model = grid_forest.best_estimator_
print(best_model)

y2_pred_train = best_model.predict(X_train_pca)
y2_pred_test = best_model.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y2_train, y2_pred_train), 
                                       mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y2_train, y2_pred_train),
                                       r2_score(y2_test, y2_pred_test)))

#Y2 PCT 6MO FWD   (std)

forest = RandomForestRegressor()
params_forest = {'n_estimators':[5,25,50,100,200],'max_features':['log2', 'auto', 'sqrt'],
             'min_samples_leaf':[1,2,5,50,100,200]}

grid_forest = GridSearchCV(estimator=forest,param_grid=params_forest,
                           scoring='neg_mean_squared_error',cv=10,verbose=1,n_jobs=-1)
grid_forest.fit(X_train_std,y2_train)
best_model = grid_forest.best_estimator_
print(best_model)

y2_pred_train = best_model.predict(X_train_std)
y2_pred_test = best_model.predict(X_test_std)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y2_train, y2_pred_train), 
                                       mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y2_train, y2_pred_train),
                                       r2_score(y2_test, y2_pred_test)))

#Y3 PCT 9MO FWD  (PCA)

forest = RandomForestRegressor()
params_forest = {'n_estimators':[5,25,50,100,200],'max_features':['log2', 'auto', 'sqrt'],
             'min_samples_leaf':[1,2,5,50,100,200]}

grid_forest = GridSearchCV(estimator=forest,param_grid=params_forest,
                           scoring='neg_mean_squared_error',cv=10,verbose=1,n_jobs=-1)
grid_forest.fit(X_train_pca,y3_train)
best_model = grid_forest.best_estimator_
print(best_model)

y3_pred_train = best_model.predict(X_train_pca)
y3_pred_test = best_model.predict(X_test_pca)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y3_train, y3_pred_train), 
                                       mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y3_train, y3_pred_train),
                                       r2_score(y3_test, y3_pred_test)))

#Y3 PCT 9MO FWD  (std)

forest = RandomForestRegressor()
params_forest = {'n_estimators':[5,25,50,100,200],'max_features':['log2', 'auto', 'sqrt'],
             'min_samples_leaf':[1,2,5,50,100,200]}

grid_forest = GridSearchCV(estimator=forest,param_grid=params_forest,
                           scoring='neg_mean_squared_error',cv=10,verbose=1,n_jobs=-1)
grid_forest.fit(X_train_std,y3_train)
best_model = grid_forest.best_estimator_
print(best_model)

y3_pred_train = best_model.predict(X_train_std)
y3_pred_test = best_model.predict(X_test_std)

print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y3_train, y3_pred_train), 
                                       mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y3_train, y3_pred_train),
                                       r2_score(y3_test, y3_pred_test)))

# Gradient Boosting
# Import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error as MSE

# Instantiate sgbr
sgbr = GradientBoostingRegressor(max_depth=4, 
            subsample=1,
            max_features=0.75,
            n_estimators=16)
# y1

# Fit sgbr to the training set
sgbr.fit(X_train_std,y1_train)
# Predict test set labels
y1_pred_train = sgbr.predict(X_train_std)
y1_pred_test = sgbr.predict(X_test_std)

# Print
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y1_train, y1_pred_train), 
                                       mean_squared_error(y1_test, y1_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y1_train, y1_pred_train),
                                       r2_score(y1_test, y1_pred_test)))

# y2

# Fit sgbr to the training set
sgbr.fit(X_train_std,y2_train)
# Predict test set labels
y2_pred_train = sgbr.predict(X_train_std)
y2_pred_test = sgbr.predict(X_test_std)

# Print
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y2_train, y2_pred_train), 
                                       mean_squared_error(y2_test, y2_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y2_train, y2_pred_train),
                                       r2_score(y2_test, y2_pred_test)))

# y3

# Fit sgbr to the training set
sgbr.fit(X_train_std,y3_train)
# Predict test set labels
y3_pred_train = sgbr.predict(X_train_std)
y3_pred_test = sgbr.predict(X_test_std)

# Print
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y3_train, y3_pred_train), 
                                       mean_squared_error(y3_test, y3_pred_test)))
print('R^2 train: %.3f, test: %.3f' % (r2_score(y3_train, y3_pred_train),
                                       r2_score(y3_test, y3_pred_test)))
