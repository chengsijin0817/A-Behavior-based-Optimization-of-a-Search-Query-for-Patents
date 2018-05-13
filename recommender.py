# -*- coding: utf-8 -*-
"""
Created on Tue May  8 13:52:17 2018

@author: chengsijin
"""
import graphlab as gl
import pandas as pd
import numpy as np
import nltk
import re
import heapq
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.text import Text
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def model(user_id = [], patent_id = [], ranking = [], factor = []):
    
    #新的专利来了之后，添加到新的结果集
    
    """
    #input title feature metrix
    t_factor_original = pd.read_csv('titleresult.csv', na_values=['NA'])
    t_factor = t_factor_original.fillna(0)
    #input abstract feature metrix
    a_factor_original = pd.read_csv('abstractresult.csv', na_values=['NA'])
    a_factor = a_factor_original.fillna(0)
    #get the all feature metrix by merge
    factor = pd.merge(t_factor, a_factor, on = 'item_id', suffixes = ('_t', '_a'), 
                      how = 'outer')
    
    
    #factor.to_csv('factor.csv')
    """
    
    #input
    sf = gl.SFrame({'user_id': user_id,
                    'item_id': patent_id,
                    'rating': ranking})                
    item_info = gl.SFrame(factor)
    
    #train, test = gl.recommender.util.random_split_by_user(sf)
    #create recommender model
    m = gl.factorization_recommender.create(sf, target='rating', 
                                             item_data=item_info)
    return m
    #ite = m.get_similar_items()
    #ite.export_csv('simi_item.csv')
    #print(ite)
def recom(m):
    result = m.recommend()
    return result                          
    #print(result)          
def predict(m, test):
    result = m.predict(test)
    return result
    #m2 = m1.evaluate_precision_recall(test)
def precision(m, test):
    m2 = m.evaluate_precision_recall(test)
    return m2
    