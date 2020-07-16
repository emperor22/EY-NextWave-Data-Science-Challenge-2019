#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:18:24 2019

@author: YAO
"""

import pandas as pd
r = pd.read_csv("data_test.csv")
r1=pd.DataFrame(r)
#delete other columns
r2=r1.drop(['trajectory_id', 'time_entry', 'time_exit','vmax','vmin','vmean','x_entry','y_entry','x_exit','y_exit'], axis=1)

#delete duplicate
r3=r2.drop_duplicates(subset='hash', keep='last', inplace=False)

#add one conlume
r3['city_center']='1'

#rename
r3 = r3.rename(columns={'hash': 'trajectory_id'})

#define r_0 and r_1: all 0 and all 1
#drop 'hash'
r_1=r_1.drop(['hash'], axis=1)
r_0=r_0.drop(['hash'], axis=1)

 
#rename
r_1.columns=['id','target']

#write to csv
r_0.to_csv("submission_0.csv",index=False,sep=',')
r_1.to_csv("submission_1.csv",index=False,sep=',')

