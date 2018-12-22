import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score
from math import log as log
import os
# print(os.listdir("../input"))
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
import math
df_train.head()

pred=df_test.restaurant_rating_value
for i in range(len(pred)):
    if not (pred[i]>=0 and pred[i]<=5):
        pred[i]=3
    # print pred[i]
result = pd.DataFrame(pred)
result.index.name = 'id'
result.columns = ['ratingValue_target']
result.to_csv('output.csv', index=True)