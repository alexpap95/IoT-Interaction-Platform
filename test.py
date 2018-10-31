# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 13:46:17 2018

@author: DELL
"""


from numpy import genfromtxt
import numpy as np

data = genfromtxt('data.dat', delimiter=',')
data_x = data[:, 2:12]
print (data_x)

#200
#250000
#100
