#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:24:20 2018

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
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


"""
#从原始搜索结果中导出12月份的专利
original_patent = pd.read_csv('Portfolio Optimizer.csv', encoding='latin-1', na_values=['NA'])


patent_clean = original_patent.loc[:,['Title', 'Abstract', 'Publication', 'IPC','Ranking']]
patent_info = patent_clean.dropna(axis = 0, how = 'any')

pattern = '2017-12'
patent_Dec = patent_info[patent_info['Publication'].str.contains(pattern)]
patent_Dec.to_csv('patentinfo_Dec.csv')

"""
#初始化搜索语句，并导入带有feedback的12月份的搜索结果
query = ["plate heat exchanger"]
patentinfo = pd.read_csv('cheng_rank_patentinfo_Dec1.csv', encoding='latin-1', na_values=['NA'])

#只留下想要的列信息，如果这一行有任何一个为空，则丢掉该行
patent_clean = patentinfo.loc[:,['Title', 'Abstract', 'Ranking']]
patent_info = patent_clean.dropna(axis = 0, how = 'any')
#patent_info.to_csv('patentinfo22.csv')

#把文件按列分析
title = patent_info['Title']
abstract = patent_info['Abstract']

patent_rank1 = patent_info[patent_info['Ranking'] == 1]
patent_rank3 = patent_info[patent_info['Ranking'] == 3]
patent_rank5 = patent_info[patent_info['Ranking'] == 5]

r = patent_info.shape[0]
r1 = patent_rank1.shape[0]
r3 = patent_rank3.shape[0]
r5 = patent_rank5.shape[0]

print(r,r1,r3,r5)

"""
alpha= 1
a= 1/r5
b = 1/r3
c= 1/r1
"""
alpha= 0
a= 1/r5
b = 1/r3
c= 1/r1


beta = a/(a+b+c)
lamda = b/(a+b+c)
gamma = -c/(a+b+c)


print(alpha, beta, lamda, gamma)



#patent_rank1.to_csv('rank1.csv')

#分词, 去停用词，去标点，stem
def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    #tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    english_stopwords = stopwords.words("english")
    self_stopwords = ['comprises','comprising','first','second','includes','use','plurality','device','structure','arranged','connected','invention','provided']
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
    st = PorterStemmer()
    words_clear=[]
    
    
    for i in tokens:
        if i.lower() not in english_stopwords: # 过滤停用词
            if i not in self_stopwords:
                if i not in english_punctuations: # 过滤标点符号
                    if re.search('[a-z]', i):
                        t = st.stem(i) 
                        words_clear.append(t)
    words_text=Text(words_clear) 
    return words_text

def tokenize_only(text):
    tokens = nltk.word_tokenize(text)
    english_stopwords = stopwords.words("english")
    self_stopwords = ['comprises','comprising','first','second','includes','use','plurality','device','structure','arranged','connected','invention','provided']
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
    
                            
    words_clear=[]
    for i in tokens:
        if i.lower() not in english_stopwords: # 过滤停用词
            if i not in self_stopwords:
                if i not in english_punctuations: # 过滤标点符号
                    if re.search('[a-z]', i):
                        words_clear.append(i)
    words_text=Text(words_clear)
    
    return words_text

"""
totalvocab_stemmed = []
totalvocab_tokenized = []

for i in abstract:
    allwords_stemmed = tokenize_and_stem(i)   
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)
#为简介中的所有词建立倒排索引
inv_indx = {i:[] for i in totalvocab_stemmed}
for word in totalvocab_stemmed:
    for i in range(len(abstract)):
        if word in abstract[i]:
            inv_indx[word].append(i)
print(inv_indx)
"""
#tfidf 向量原型
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
#------------------------对标题进行预处理-----------------------------
#tfidf_matrix = tfidf_vectorizer.fit_transform(patent_info['Title'].values.astype('U'))
#print(tfidf_matrix)
titleVectorizerArray = tfidf_vectorizer.fit_transform(patent_info['Title'].values.astype('U')).toarray()
t_queryVectorizerArray = tfidf_vectorizer.transform(query).toarray()
#print('Title:',t_queryVectorizerArray)
#tf-idf 矩阵中的特征（features）表
terms = tfidf_vectorizer.get_feature_names()
#print(terms)
titledf = pd.DataFrame(titleVectorizerArray, columns= [terms])
qyertdf = pd.DataFrame(t_queryVectorizerArray, columns= [terms])

#titledf.to_csv('titleVector.csv')

#ar1 = titledf.loc[patent_rank1.index,:]
#ar1.to_csv('titlerank1.csv')
#ar5 = abstractdf.loc[patent_rank5.index,:]
#-----Rocchio
"""
alpha=1
beta=0.8
lamda = 0.1
gamma= -0.1
"""

qu  = alpha*qyertdf
ti5 = beta*titledf.loc[patent_rank5.index,:]
ti3 = lamda*titledf.loc[patent_rank3.index,:]
ti1 = gamma*titledf.loc[patent_rank1.index,:]

t_res = pd.concat([qu, ti5, ti3, ti1], axis = 0)
#t_res.to_csv('titleresult.csv')

t_wei = np.sum(t_res, axis = 0)
print(t_wei)

#输出最大权重的下标
#idx = np.argmax(t_wei, axis=1)

z = heapq.nlargest(8,t_wei)
#print(z)
wordlist = []
for i in t_wei.index:
    for j in z:
        if(t_wei[i] == j):
            wordlist.append(i)
print('@ti',wordlist)

#------------------------对简介进行预处理-----------------------------
#tfidf_matrix = tfidf_vectorizer.fit_transform(patent_info['Abstract'].values.astype('U'))
#print(tfidf_matrix

abstractVectorizerArray = tfidf_vectorizer.fit_transform(patent_info['Abstract'].values.astype('U')).toarray()
queryVectorizerArray = tfidf_vectorizer.transform(query).toarray()
#print('Abstract:',queryVectorizerArray)
#tf-idf 矩阵中的特征（features）表
terms = tfidf_vectorizer.get_feature_names()
#print(terms)
abstractdf = pd.DataFrame(abstractVectorizerArray, columns= [terms])
qyertdf = pd.DataFrame(queryVectorizerArray, columns= [terms])

#abstractdf.to_csv('AbstractVector.csv')

#ar5 = abstractdf.loc[patent_rank5.index,:]


qu  = alpha*qyertdf
ar5 = beta*abstractdf.loc[patent_rank5.index,:]
ar3 = lamda*abstractdf.loc[patent_rank3.index,:]
ar1 = gamma*abstractdf.loc[patent_rank1.index,:]

res = pd.concat([qu, ar5, ar3, ar1], axis = 0)
#res.to_csv('abstractresult.csv')

wei = np.sum(res, axis = 0)
print(wei)

#输出最大权重的下标
idx = np.argmax(wei, axis=1)

z = heapq.nlargest(8,wei)
#print(z)
wordlist = []
for i in wei.index:
    for j in z:
        if(wei[i] == j):
            wordlist.append(i)
print('@ab',wordlist)
#---------------------------------------------------------------------------------------------

