# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:46:17 2018

@author: DELL
"""


import pandas as pd

df = pd.read_csv("data.csv", dtype=float)
df = df[(df['1'])!=1.0]
df.to_csv("newdata.csv", header=False, index=False)

#data = genfromtxt('test.dat', delimiter=',')
#data_x = data[:, 2:12]
#print (data_x)

#200
#250000
#100
