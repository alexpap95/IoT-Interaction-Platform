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

col_names=['ID','Time','accx','accy','accz','gyrox','gyroy','gyroz','q1','q2','q3','q4']
df = pd.read_csv('5_11.csv', names=col_names, header=None)
print (df['ID'].value_counts())
#print (df[df['ID']==2.0])
#df.drop(df[df['ID']==3.0].index[:3000], inplace=True)
#df.drop(df[df['ID']==1.0].index[:3000], inplace=True)
#df.drop(df[df['ID']==2.0].index[:3000], inplace=True)
#df.drop(df[df['ID']==4.0].index[:3000], inplace=True)
#df.to_csv('5_11.csv',index=None,header=None)


        