# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:32:31 2018

@author: chengsijin
"""

import pandas as pd
import numpy as np
import nltk
import re
import heapq
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.text import Text
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_param(patent1 = [], patent3 = [], patent5 = []):
    
    r1 =len(patent1)
    r3 =len(patent3)
    r5 =len(patent5)
    
    print(r1, r3, r5)
    alpha= 0
    if r5 !=0:
        a= 1.0/r5
    else:
        r5 = 1
        a= 1.0/r5
    if r3 !=0:
        b = 1.0/r3
    else:
        r3 = 1
        b = 1.0/r3
    if r1!=0:
        c= 1.0/r1
    else:
        r1 = 1
        c= 1.0/r1
    
    beta = a/(a+b+c)
    lamda = b/(a+b+c)
    gamma = -c/(a+b+c)      
    return alpha, beta, lamda, gamma
    
def rocchio(vector = [], patent1 = [], patent3 = [], patent5 = []):
    # Rocchio algorithm: consider different rank
    #qu  = alpha*qyertdf
    t = calculate_param(patent1, patent3, patent5)
    print('alpha, beta, lamda, gamma:', t)
    ar5 = t[1]*vector.loc[patent5,:]
    ar3 = t[2]*vector.loc[patent3,:]
    ar1 = t[3]*vector.loc[patent1,:]
    
    a_res = pd.concat([ar5, ar3, ar1], axis = 0)
    #a_res.to_csv('abstractresult.csv')    
    a_wei = np.sum(a_res, axis = 0)
    print('weight:', a_wei)
    #输出最大权重的下标
    
    #the index of the highest sum_weight
    #idx = np.argmax(a_wei, axis=1)
    
    #print the top 8 keywords to form the new query
    z = heapq.nlargest(8,a_wei)
    awordlist = []
    for i in a_wei.index:
        for j in z:
            if(a_wei[i] == j):
                awordlist.append(i)
    return awordlist