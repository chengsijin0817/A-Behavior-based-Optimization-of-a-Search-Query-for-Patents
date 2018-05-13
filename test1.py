# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:49:07 2018

@author: chengsijin
"""

import preprocess
import pymysql.cursors
from pandas import Series
from collections import Counter
import Rocchio

patent_id = []
row_title =[]
row_abstract =[]

patent1 = []
patent3 = []
patent5 = []

st1 = '%2017-12%'
st2 = '%2018-01%'
st3 = '%2018-02%'
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='patentRecom',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "select r.`patent_id`, i.`title`, i.`abstract` from `patent_info` i join `patent_rank` r on i.`patent_id`=r.`patent_id` where (i.`publication` like %s or i.`publication` like %s or i.`publication` like %s) and i.`query` = %s"
        cursor.execute(sql, (st1, st2, st3, 1, ))     
        for result in cursor.fetchall():
            patent_id.append(result['patent_id'])
            row_title.append(result['title'])
            row_abstract.append(result['abstract'])
    with connection.cursor() as cursor:
        # Read a single record
        sql = "select r.`patent_id` from `patent_info` i join `patent_rank` r on i.`patent_id`=r.`patent_id` where (i.`publication` like %s or i.`publication` like %s or i.`publication` like %s) and i.`query` = %s and r.`ranking` = %s"
        cursor.execute(sql, (st1, st2, st3, 1, 5, ))     
        for result in cursor.fetchall():
            patent5.append(result['patent_id'])
    with connection.cursor() as cursor:
        # Read a single record
        sql = "select r.`patent_id` from `patent_info` i join `patent_rank` r on i.`patent_id`=r.`patent_id` where (i.`publication` like %s or i.`publication` like %s or i.`publication` like %s) and i.`query` = %s and r.`ranking` = %s"
        cursor.execute(sql, (st1, st2, st3, 1, 3, ))  
        for result in cursor.fetchall():
            patent3.append(result['patent_id'])
    with connection.cursor() as cursor:
        # Read a single record
        sql = "select r.`patent_id` from `patent_info` i join `patent_rank` r on i.`patent_id`=r.`patent_id` where (i.`publication` like %s or i.`publication` like %s or i.`publication` like %s) and i.`query` = %s and r.`ranking` = %s"
        cursor.execute(sql, (st1, st2, st3, 1, 1, ))  
        for result in cursor.fetchall():
            patent1.append(result['patent_id'])
finally:
    connection.close()

title = Series(row_title, index = patent_id)
t_vector = preprocess.tfidf(title)
#t_vector.index.name = 'item_id'

title_query= Rocchio.rocchio(t_vector, patent1, patent3, patent5)
print('@ti', title_query)

abstract = Series(row_abstract, index = patent_id)
ab_vector = preprocess.tfidf(abstract)

abstract_query = Rocchio.rocchio(ab_vector, patent1, patent3, patent5)
print('@ab', abstract_query)
#print(ab_vector)



