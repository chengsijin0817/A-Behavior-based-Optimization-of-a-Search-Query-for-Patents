#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:24:20 2018

@author: chengsijin
"""

#print(patent_info[0:3])


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.text import Text
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
import gensim  


#import *xlsx file
xlsx = pd.ExcelFile('Portfolio Optimizer.xlsx')
patent_info = pd.read_excel(xlsx, 'Optimizer', na_values=['NA'])

rank1 = patent_info[patent_info['Ranking'] == 'Main ranking: 1\n']
rank3 = patent_info[patent_info['Ranking'] == 'Main ranking: 3\n']
rank5 = patent_info[patent_info['Ranking'] == 'Main ranking: 5\n']

IPC1 = rank1['IPC'].value_counts()
IPC3 = rank3['IPC'].value_counts()
IPC5 = rank5['IPC'].value_counts() 

#print(IPC1)
#print(IPC3)
#print(IPC5)
#import *csv file
#patent_info = pd.read_csv('Optimizer.csv', encoding='latin-1',na_values=['NA'])

"""
#或许这里根本不需要建这么多数组，只需要知道相应的索引号就可以了
title_a = rank1['Patent Title']
abstract_a = rank1['Abstract']
title_b = rank3['Patent Title']
abstract_b = rank3['Abstract']
title_c = rank5['Patent Title']
abstract_C = rank5['Abstract']
"""
title = patent_info['Patent Title']
abstract = patent_info['Abstract']

#print(abstract_a)
#print(abstract)
#print(title)
#print(abstract[100])

#tokenize

def tokenize_and_stem(text):
    tokens = nltk.word_tokenize(text)
    #tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    
    english_stopwords = stopwords.words("english")
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
    st = PorterStemmer()
    words_clear=[]
    
    
    for i in tokens:
        if i.lower() not in english_stopwords: # 过滤停用词
            if i not in english_punctuations: # 过滤标点符号
                if re.search('[a-z]', i):
                    t = st.stem(i) 
                    words_clear.append(t)
    words_text=Text(words_clear) 
    return words_text
 


def tokenize_only(text):
    tokens = nltk.word_tokenize(text)
    english_stopwords = stopwords.words("english")
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
    
                            
    words_clear=[]
    for i in tokens:
        if i.lower() not in english_stopwords: # 过滤停用词
            if i not in english_punctuations: # 过滤标点符号
                if re.search('[a-z]', i):
                    words_clear.append(i)
    words_text=Text(words_clear)
    
    return words_text  
"""
totalvocab_stemmed = []
totalvocab_tokenized = []

for i in range(50):
    allwords_stemmed = tokenize_and_stem(abstract[i])   
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(abstract[i])
    totalvocab_tokenized.extend(allwords_tokenized)


words_counter=Counter(totalvocab_stemmed) 
#print(words_counter)

#词干化后的词和原词构成了一个对比词表
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print(vocab_frame.head())
"""
"""
#tfidf 向量原型
tfidf_vectorizer1_5 = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_only, ngram_range=(1,3))
tfidf_vectorizer1_1 = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_only, ngram_range=(1,3))
tfidf_vectorizer2 = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
 
tfidf_matrix1_5 = tfidf_vectorizer1_5.fit_transform(rank5['Abstract'].values.astype('U'))
tfidf_matrix1_1 = tfidf_vectorizer1_1.fit_transform(rank1['Abstract'].values.astype('U'))
tfidf_matrix2 = tfidf_vectorizer2.fit_transform(rank5['Abstract'].values.astype('U'))
#print(tfidf_matrix)


#tf-idf 矩阵中的特征（features）表
terms1_1 = tfidf_vectorizer1_5.get_feature_names()
terms1_5 = tfidf_vectorizer1_1.get_feature_names()
terms2 = tfidf_vectorizer2.get_feature_names()
#print(terms1_1)
#print(terms1_5)

#print(terms2)

dist = 1 - cosine_similarity(tfidf_matrix)
print(dist)


#clustering
from sklearn.cluster import KMeans
 
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print(clusters)
"""

texts = []
# loop through document list  
for i in patent_info.index:    
    allwords_stemmed = tokenize_and_stem(abstract[i])
    texts.append(allwords_stemmed)
 
# turn our tokenized documents into a id <-> term dictionary  
dictionary = corpora.Dictionary(texts)  
      
# convert tokenized documents into a document-term matrix  
corpus = [dictionary.doc2bow(text) for text in texts]  
  
# generate LDA model  
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=3, num_words=10))



