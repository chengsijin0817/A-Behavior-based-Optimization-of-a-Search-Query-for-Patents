# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:05:27 2018

@author: chengsijin
"""

import pandas as pd
from sqlalchemy import create_engine

original_patent = pd.read_csv('Portfolio Optimizer.csv', encoding='latin-1', na_values=['NA'])

patent = original_patent.loc[:,['patent_id', 'title', 'abstract', 'publication', 'query']]

##将数据写入mysql的数据库，但需要先通过sqlalchemy.create_engine建立连接,且字符编码设置为utf8，否则有些latin字符不能处理  
yconnect = create_engine('mysql+pymysql://root:@localhost:3306/patentRecom?charset=utf8')  

#patent信息表
pd.io.sql.to_sql(patent,'patent_info', yconnect, schema='patentRecom', if_exists='append',index=False)

#用户数据表
user = pd.read_csv('user.csv', encoding='latin-1', na_values=['NA'])
pd.io.sql.to_sql(user,'user_info', yconnect, schema='patentRecom', if_exists='append',index=False)

#排名表
rank = original_patent.loc[:,['user_id', 'patent_id', 'ranking']]
pd.io.sql.to_sql(rank,'patent_rank', yconnect, schema='patentRecom', if_exists='append',index=False)
