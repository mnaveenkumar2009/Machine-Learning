import nltk
from nltk.classify import NaiveBayesClassifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from math import log as log
import os
print(os.listdir("data"))
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def fn(A):
    A['xxx']=A['ba']
    A['xxx2']=A['baaa']
    A['xx3']=A['baa']
    # A['aaa']
    return A
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
# (df_train.replace(r'', 'xxx', regex=True))

import math
df_train.head()
df_test.head()
df_train.ratingValue_target-=1
train_X = df_train.loc[0:31999, ['restaurant_rating_value','restaurant_rating_atmosphere','restaurant_rating_food']]
train_y = df_train.loc[0:31999, 'ratingValue_target']
# train_y['ratingValue_target']=train_y['ratingValue_target']-1
train_X['a']=1.3
train_X['aa']=1.3
train_X['aaa']=1.3
train_X['aaaa']=1.3
train_X['b']=1.3
train_X['ba']=1.3
train_X['baa']=1.3
train_X['baaa']=1.3
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# input("sentiment")
sid = SentimentIntensityAnalyzer()
sentences=df_train.loc[0:31999, ['text']]
sentences2=df_train.loc[0:31999, ['title']]
print("ggre")
print (sentences2['title'][0])

for j in range(32000):
    # print(j)
    sentence=sentences2['title'][j]
    if(sentence!=sentence):
        sentence="xxx"
    ss = sid.polarity_scores(sentence)
    i='a'
    for k in sorted(ss):
#         print('{0}: {1}, '.format(k, ss[k]))
        # print (k)
        # print(ss[k])
        train_X[i][j]=ss[k]
        i+='a'
    if j%1000==0:
        print (j)
        # print (train_X.head())
        # print (train_X['a'][1])
        # print (train_X['aa'][1])
        # print (train_X['aaa'][1])
        # print (train_X['aaaa'][1])
for j in range(32000):
    # print(j)
    sentence=sentences['text'][j]
    if(sentence!=sentence):
        sentence="xxx"
    ss = sid.polarity_scores(sentence)
    i='b'
    for k in sorted(ss):
#         print('{0}: {1}, '.format(k, ss[k]))
        # print (k)
        # print(ss[k])
        train_X[i][j]=ss[k]
        i+='a'
    if j%1000==0:
        print (j)

train_X=fn(train_X)

data_dmatrix = xgb.DMatrix(data=train_X,label=train_y)
xg_reg = xgb.XGBRegressor(objective ='multi:softprob', num_class = 5,colsample_bytree = 0.3, learning_rate = 0.02,
                max_depth = 5, alpha = 10, n_estimators = 300,eval_metric='auc')
xg_reg.fit(train_X,train_y)


testX = df_train.loc[32000:39999, ['restaurant_rating_value','restaurant_rating_atmosphere','restaurant_rating_food']]
sentences=df_train.loc[32000:39999, ['text']]
sentences2=df_train.loc[32000:39999, ['title']]
testX['a']=1.3
testX['aa']=1.3
testX['aaa']=1.3
testX['aaaa']=1.3
testX['b']=1.3
testX['ba']=1.3
testX['baa']=1.3
testX['baaa']=1.3
for j in range(8000):
    # print(j)
    sentence=sentences2['title'][j+32000]
    if(sentence!=sentence):
        sentence="xxx"
    ss = sid.polarity_scores(sentence)
    i='a'
    for k in sorted(ss):
        testX[i][j+32000]=ss[k]
        i+='a'
        # print(k)
        # print(ss[k])
        # print(i)
    # input()


for j in range(8000):
    # print(j)
    sentence=sentences['text'][j+32000]
    if(sentence!=sentence):
        sentence="xxx"
    ss = sid.polarity_scores(sentence)
    i='b'
    for k in sorted(ss):
        testX[i][j+32000]=ss[k]
        i+='a'

testX=fn(testX)
pred = xg_reg.predict(testX)
import numpy as np
classes=[0]*8000
for i in range(8000):
    classes[i] = np.argmax(pred[i])+1
    # if(testX['aaaa'][32000+i]>0.4):
    #     classes[i]=5
    if(testX['baaa'][32000+i]>0.2):
        if not (classes[i]==5):
            classes[i]=4 
    if(testX['baaa'][32000+i]>0.4):
        classes[i]=5    
    # if(testX['aa'][32000+i]>0.4):
    #     classes[i]=1
    if(testX['ba'][32000+i]>0.4):
        classes[i]=1
        
    
print (classes)
from sklearn.metrics import f1_score
y_true = df_train.loc[32000:39999, 'ratingValue_target']
y_true+=1
print(f1_score(y_true, classes, average='macro')  )
print(f1_score(y_true, classes, average='micro'))
print(f1_score(y_true, classes, average='weighted')  )
print(f1_score(y_true, classes, average=None))




# Test  data

testX = df_test.loc[0:19999, ['restaurant_rating_value','restaurant_rating_atmosphere','restaurant_rating_food']]
sentences=df_test.loc[0:19999, ['text']]
sentences2=df_test.loc[0:19999, ['title']]
testX['a']=1.3
testX['aa']=1.3
testX['aaa']=1.3
testX['aaaa']=1.3
testX['b']=1.3
testX['ba']=1.3
testX['baa']=1.3
testX['baaa']=1.3


for j in range(20000):
    sentence=sentences2['title'][j]
    if(sentence!=sentence):
        sentence="xxx"
    ss = sid.polarity_scores(sentence)
    i='a'
    for k in sorted(ss):
        testX[i][j]=ss[k]
        i+='a'


for j in range(20000):
    sentence=sentences['text'][j]
    if(sentence!=sentence):
        sentence="xxx"
    ss = sid.polarity_scores(sentence)
    i='b'
    for k in sorted(ss):
        testX[i][j]=ss[k]
        i+='a'
testX=fn(testX)
pred = xg_reg.predict(testX)

classes=[0]*20000
for i in range(20000):
    classes[i] = np.argmax(pred[i])+1
    # if(testX['aaaa'][i]>0.4):
    #     classes[i]=5
    if(testX['baaa'][i]>0.4):
        classes[i]=5    

    if(testX['baaa'][i]>0.2):
        if not (classes[i]==5):
            classes[i]=4
    # if(testX['aa'][i]>0.4):
    #     classes[i]=1
    if(testX['ba'][i]>0.4):
        classes[i]=1

result = pd.DataFrame(classes)
result.index.name = 'id'
result.columns = ['ratingValue_target']
result.to_csv('output.csv', index=True)