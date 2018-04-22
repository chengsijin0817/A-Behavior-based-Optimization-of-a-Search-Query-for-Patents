# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 18:23:51 2018

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



patentinfo = pd.read_csv('Dec_Rec_input_patent.csv', encoding='latin-1', na_values=['NA'])
patentinfo['user']= '1'


#只留下想要的列信息，如果这一行有任何一个为空，则丢掉该行
patent_clean = patentinfo.loc[:,['user', 'Title', 'Abstract', 'Ranking']]
patent_info = patent_clean.dropna(axis = 0, how = 'any')
#patent_info.to_csv('patentinfo22.csv')
#print(patent_info)

 # create array to store data as input of cf_table
user_id = patent_info['user']
patent_id = patent_info.index
ranking =patent_info['Ranking']
title = patent_info['Title']
abstract = patent_info['Abstract']

#print(patent_id)

#新的专利来了之后，添加到新的结果集



t_factor_original = pd.read_csv('titleresult.csv', na_values=['NA'])
t_factor = t_factor_original.fillna(0)
#print(factor)



#factor.fillna('0')
patent= t_factor['id']
t_exchang= t_factor['exchang']
t_heat = t_factor['heat']
t_heat_exchang = t_factor['heat exchang']

a_factor_original = pd.read_csv('abstractresult.csv', na_values=['NA'])
a_factor = a_factor_original.fillna(0)
a_air = a_factor['air']
a_bodi = a_factor['bodi']
a_chamber = a_factor['chamber']
a_commun = a_factor['commun']
a_cool = a_factor['cool']
a_direct = a_factor['direct']
a_disclos = a_factor['disclos']
a_effici = a_factor['effici']
a_extend = a_factor['extend']
a_flow = a_factor['flow']
a_form = a_factor['form']
a_heat_exchang = a_factor['heat exchang']
a_inlet = a_factor['inlet']
a_outlet = a_factor['outlet']
a_plate = a_factor['plate']
a_plate_heat = a_factor['plate heat']
a_surfac = a_factor['surfac']
a_unit = a_factor['unit']
a_wall = a_factor['wall']
a_water = a_factor['water']

sf = gl.SFrame({'user_id': user_id,
                'item_id': patent_id,
                'rating': ranking})

item_info = gl.SFrame({'item_id': patent,
                        't_exchang':t_exchang,
                        't_heat': t_heat,
                        't_heat exchang':t_heat_exchang,
                        'a_air': a_air,
                        'a_bodi': a_bodi,
                        'a_chamber': a_chamber,
                        'a_commun': a_commun,
                        'a_cool': a_cool,
                        'a_direct':a_direct,
                        'a_disclos':a_disclos,
                        'a_effici':a_effici,
                        'a_extend':a_extend,
                        'a_flow':a_flow,
                        'a_form':a_form,
                        'a_heat_exchang':a_heat_exchang,
                        'a_inlet': a_inlet,
                        'a_outlet': a_outlet,
                        'a_plate': a_plate,
                        'a_plate_heat': a_plate_heat,
                        'a_surfac': a_surfac,
                        'a_unit': a_unit,
                        'a_wall': a_wall,
                        'a_water': a_water
                      })

#train, test = gl.recommender.util.random_split_by_user(sf)
m1 = gl.factorization_recommender.create(sf, target='rating', item_data=item_info)

#m2 = m1.evaluate_precision_recall(test)
print(m1.recommend())
print(m1.get_num_items_per_user())