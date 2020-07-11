#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:18:24 2019

@author: YAO
"""

import numpy as np # linear algebra
import pandas as pd

import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)

from scipy import stats
from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))

train = pd.read_csv('data_train.csv')
test = pd.read_csv('data_test.csv')

train1=train.drop(columns=['Unnamed: 0'], axis=1)
test1=test.drop(columns=['Unnamed: 0'], axis=1)

tra=pd.DataFrame(train1)
tst=pd.DataFrame(test1)
i=0
for row in tra.itertuples(index=True, name='train'):
    print getattr(row, "c1"), getattr(row, "c2")
    
tst.ix[i:i,[0]]==tst.ix[i:i,[0]]


tra['sameTime']=sameTime

for i in range(0,100):
    i=i+1
    if (tra['time_entry'] == tra['time_exit']).all():
        print(i)

for row in tra.itertuples(index=True, name='train'):
    if tra.ix['time_entry']== tra.ix['time_exit']:
        i=i+1
        print(i)
        
e=[]
tra1=tra
i=0
for row in tra.itertuples(index=True, name='Pandas'):
    i=i+1
    if getattr(row, "sameTime"):
        e.append(i)
E={"detect":e}

        
