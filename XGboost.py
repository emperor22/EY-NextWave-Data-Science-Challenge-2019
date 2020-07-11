#!/usr/bin/env python
# coding: utf-8

# In[14]:


import xgboost as xgb
import numpy as np
import pandas as pd
train = pd.read_csv("./pytrain.csv", index_col = 0)
test = pd.read_csv(" ./pytest.csv", index_col = 0)


# In[15]:


X = np.array(train.ix[:,0:311])
y = np.array(train['SalePrice'])
test = np.array(test)


# In[3]:


from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
dtrain = xgb.DMatrix(X, label = y)
params = {"max_depth":2, "eta":0.1}
XGB = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)


# In[5]:


XGB.loc[100:,["test-rmse-mean", "train-rmse-mean"]].plot()
# can find accuray about 0.125


# In[12]:


XGBOOST = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
XGBOOST.fit(X, y)
predict = np.expm1(XGBOOST.predict(test))


# In[10]:


from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
predict = 1 + np.zeros(1459)
kf = KFold(n_splits=20,random_state=666,shuffle=True)
test_errors = []
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    XGBOOST = make_pipeline(RobustScaler(), xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.125)) #the params were tuned using xgb.cv
    XGBOOST.fit(Xtrain, ytrain)
    pred = XGBOOST.predict(test)
    predict = predict * pred
    xpre = XGBOOST.predict(Xtest)
    test_errors.append(np.square(xpre - ytest).mean() ** 0.5)
predict = np.expm1(predict ** (1/20))
print(np.mean(test_errors))

# In[17]:

from pandas.core.frame import DataFrame
result = DataFrame(predict)
result.to_csv('finalxgboost.csv')

