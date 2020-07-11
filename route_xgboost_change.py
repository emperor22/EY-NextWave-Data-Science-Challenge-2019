import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from pandas.core.frame import DataFrame

train = pd.read_csv("./data/tra1.csv", index_col = 0)
test = pd.read_csv("./data/test.csv")

X = np.array(train.ix[:,0:3])
y = np.array(train['center'])
test_id = test.ix[:,0]
test = np.array(test.ix[:,1:4])

#
# # 矩阵化
# dtrain = xgb.DMatrix(X, label = y)
# params = {"max_depth":2, "eta":0.1}
# XGB = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
#
# XGB.loc[100:,["test-rmse-mean", "train-rmse-mean"]].plot()
# # can find accuray about 0.125
#
# # 训练
# XGBOOST = xgb.XGBClassifier(n_estimators=360, max_depth=5, learning_rate=0.1, objective='binary:logitraw') #the params were tuned using xgb.cv
# XGBOOST.fit(X, y)
# predict = XGBOOST.predict(test)

predict = np.zeros(33515)
kf = KFold(n_splits=20,random_state=666,shuffle=True)
test_errors = []
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    XGBOOST = make_pipeline(RobustScaler(), xgb.XGBClassifier(n_estimators=360, max_depth=5, learning_rate=0.125, objective='binary:logitraw')) #the params were tuned using xgb.cv
    XGBOOST.fit(Xtrain, ytrain)
    pred = XGBOOST.predict(test)
    predict = predict + pred
    # xpre = XGBOOST.predict(Xtest)
    # test_errors.append(np.square(xpre - ytest).mean() ** 0.5)
predict = predict/20.0
# print(np.mean(test_errors))


result = DataFrame(predict,test_id)
result.to_csv('xgboost4.csv')