# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:10:26 2018

@author: chengsijin
"""

import pymysql.cursors

sql1 = "DROP TABLE IF EXISTS patent_rank CASCADE;"
sql2 = "DROP TABLE IF EXISTS patent_info CASCADE;"
sql3 = "DROP TABLE IF EXISTS user_info CASCADE;"
    
sql4 = "CREATE TABLE `patent_info` (\
    `patent_id` int(11) NOT NULL AUTO_INCREMENT,\
    `title` text,\
    `abstract` text,\
    `publication` text NOT NULL,\
    `query` int(11) NOT NULL,\
    PRIMARY KEY (`patent_id`)\
    ) ENGINE=InnoDB"
sql5 = "CREATE TABLE `user_info` (\
    `user_id` int(11) NOT NULL AUTO_INCREMENT,\
    `name` varchar(32),\
    PRIMARY KEY (`user_id`)\
    ) ENGINE=InnoDB"
sql6 = "CREATE TABLE `patent_rank` (\
    `user_id` int(11) NOT NULL,\
    `patent_id` int(11) NOT NULL,\
    `ranking` int(11) NOT NULL,\
    PRIMARY KEY (`user_id`, `patent_id`),\
    CONSTRAINT `rank_patent` FOREIGN KEY (`patent_id`) REFERENCES `patent_info` (`patent_id`) ON DELETE CASCADE,\
    CONSTRAINT `rank_user` FOREIGN KEY (`user_id`) REFERENCES `user_info`(`user_id`) ON DELETE CASCADE\
    ) ENGINE=InnoDB"

# Connect to the database

connection = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             db='patentRecom',
                             cursorclass=pymysql.cursors.DictCursor)

try:
    with connection.cursor() as cursor:
        cursor.execute(sql1)
        cursor.execute(sql2)
        cursor.execute(sql3)
        cursor.execute(sql4)
        cursor.execute(sql5)
        cursor.execute(sql6)
    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()

finally:
    connection.close()