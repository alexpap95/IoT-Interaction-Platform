# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 22:22:06 2018

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 20:51:39 2018

@author: DELL
"""

import pandas as pd
import numpy as np

col_names=['ID','accx','accy','accz','gyrox','gyroy','gyroz','magx','magy','magz','q1','q2','q3','q4']
#df = pd.read_csv('full_dataset.csv', names=col_names, header=None)
#print (df['ID'].value_counts())
#df1 = df[df['ID']==4.0]
df = pd.read_csv('dataset.csv', names=col_names, header=None)
df2= df['ID']
df1=df.drop(df.columns[0], axis=1)
mu, sigma = 0, 0.001 
# creating a noise with the same dimension as the dataset (2,2) 
noise = np.random.normal(mu, sigma, [50400,13])
new = df1+noise
df3 = new.join(df2, lsuffix='_df2', rsuffix='_new')
df4 = df3[col_names]
df5=df.append(df4)
df5.to_csv('data_wnoise.csv',index=None,header=None)
#print (df['ID'].value_counts())
#df2 = df[df['ID']!=4.0]
#df = df2.append(df1)
#print (df['ID'].value_counts()/15)
#df.to_csv('red_dataset.csv', index=None, header=None)
#df2=df1.tail(7200)
#df=df2.reindex(np.random.permutation(df2.index))
#df=df1.append(df)
#print (df)
#df.to_csv('dataset.csv',index=None,header=None)
#print (df[df['ID']==2.0])
#df.drop(df[df['ID']==3.0].index[:3000], inplace=True)
#df.drop(df[df['ID']==1.0].index[:3000], inplace=True)
#df.drop(df[df['ID']==2.0].index[:3015], inplace=True)
#df.to_csv('redv2_dataset.csv',index=None,header=None)


        