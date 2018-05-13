# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:58:26 2018

@author: chengsijin
"""
import preprocess
import pymysql.cursors
from pandas import Series
import recommender
import pandas as pd
import graphlab as gl

all_patent_id = []
row_title =[]
row_abstract =[]

user_id_h = []
rank_patent_id_h = []
ranking_h = []

user_id_p = []
rank_patent_id_p = []
ranking_p = []

hold_st1 = '%2017-12%'
hold_st2 = '%2018-01%'
hold_st3 = '%2018-02%'
predict_st = '%2018-03%'

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='patentRecom',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    
    with connection.cursor() as cursor:
        # Read a single record
        sql = "SELECT `patent_id`, `title`, `abstract` FROM `patent_info` WHERE (`publication` like %s or `publication` like %s or `publication` like %s or `publication` like %s) and `query` = %s"
        cursor.execute(sql, (hold_st1, hold_st2, hold_st3, predict_st, 1))     
        for result in cursor.fetchall():
            #print(result)
            row_title.append(result['title'])
            row_abstract.append(result['abstract'])
            all_patent_id.append(result['patent_id'])
    with connection.cursor() as cursor:
        # Read a single record
        sql = "select r.`user_id`, r.`patent_id`, r.`ranking` from `patent_info` i join `patent_rank` r on i.`patent_id`=r.`patent_id` where (i.`publication` like %s or i.`publication` like %s or i.`publication` like %s) and i.`query` = %s"
        cursor.execute(sql, (hold_st1, hold_st2, hold_st3, 1 ))     
        for result in cursor.fetchall():
            rank_patent_id_h.append(result['patent_id'])
            user_id_h.append(result['user_id'])
            ranking_h.append(result['ranking'])
    with connection.cursor() as cursor:
        # Read a single record
        sql = "select r.`user_id`, r.`patent_id`, r.`ranking` from `patent_info` i join `patent_rank` r on i.`patent_id`=r.`patent_id` where i.`publication` like %s and i.`query` = %s"
        cursor.execute(sql, (predict_st, 1 ))     
        for result in cursor.fetchall():
            rank_patent_id_p.append(result['patent_id'])
            user_id_p.append(result['user_id'])
            ranking_p.append(result['ranking'])
    
finally:
    connection.close()
"""
def query_rank(hold_st):
    try:
        with connection.cursor() as cursor:
                # Read a single record
                sql = "select r.`user_id`, r.`patent_id`, r.`ranking` from `patent_info` i join `patent_rank` r on i.`patent_id`=r.`patent_id` where i.`publication` like %s and i.`query` = %s"
                cursor.execute(sql, (hold_st, 1 ))     
                for result in cursor.fetchall():
                    rank_patent_id_h.append(result['patent_id'])
                    user_id_h.append(result['user_id'])
                    ranking_h.append(result['ranking'])
    finally:
        connection.close()
    return rank_patent_id_h, user_id_h, ranking_h
"""

# 准备推荐输入数据
abstract = Series(row_abstract, index = all_patent_id)
ab_vector = preprocess.tfidf(abstract)
ab_vector['item_id'] = all_patent_id
#print(ab_vector)
title = Series(row_title, index = all_patent_id)
t_vector = preprocess.tfidf(title)
#t_vector.index.name = 'item_id'
t_vector['item_id'] = all_patent_id
#print(t_vector)

t_factor = t_vector.fillna(0)
#input abstract feature metrix
a_factor = ab_vector.fillna(0)
#get the all feature metrix by merge
factor = pd.merge(t_factor, a_factor, on = 'item_id', suffixes = ('_t', '_a'), 
                  how = 'outer')
# 输入原始数据评分,三列
#调用推荐函数
"""
reco_model = recommender.model(user_id_h, rank_patent_id_h, ranking_h, factor)
reco_res = recommender.recom(reco_model)
reco_res.export_csv('predict_rank.csv')
print(reco_res)
"""
test = gl.SFrame({'user_id': user_id_p,
                  'item_id': rank_patent_id_p,
                  'rating': ranking_p})
#test.export_csv('test_rank.csv')

reco_model = recommender.model(user_id_h, rank_patent_id_h, ranking_h, factor)
reco_res = recommender.predict(reco_model, test)
test.add_column(reco_res, name='predict_score')
test.export_csv('predict_rank.csv')
print(reco_res)

reco_precision = recommender.precision(reco_model, test)
#test.add_column(reco_precision, name='precision')
print(reco_precision)
