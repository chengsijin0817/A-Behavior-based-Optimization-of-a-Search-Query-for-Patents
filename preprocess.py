
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:24:20 2018

@author: chengsijin
"""

import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.text import Text
from sklearn.feature_extraction.text import TfidfVectorizer


#分词, 去停用词，去标点，stem

def tokenize_and_stem(text):
    #tokenize
    tokens = nltk.word_tokenize(text)
    #remove stopwords
    english_stopwords = stopwords.words("english")
    self_stopwords = ['comprises','comprising','first','second','includes',
                      'use','plurality','device','structure','arranged',
                      'connected','invention','provided', 'receive', 'provide',
                      'extend', 'relates', 'configured','method']
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!',
                            '@', '#', '%', '$', '*']
    #stem
    st = PorterStemmer()
    words_clear=[]  
    
    for i in tokens:
        if i.lower() not in english_stopwords:
            if i not in self_stopwords:
                if i not in english_punctuations:
                    if re.search('[a-z]', i):
                        t = st.stem(i) 
                        words_clear.append(t)
    words_text=Text(words_clear) 
    return words_text

def tokenize_only(text):
    tokens = nltk.word_tokenize(text)
    english_stopwords = stopwords.words("english")
    self_stopwords = ['comprises','comprising','first','second','includes',
                      'use','plurality','device','structure','arranged',
                      'connected','invention','provided', 'receive', 'provide',
                      'extend', 'relates', 'configured', 'method']
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
    
                            
    words_clear=[]
    for i in tokens:
        if i.lower() not in english_stopwords:
            if i not in self_stopwords:
                if i not in english_punctuations: 
                    if re.search('[a-z]', i):
                        words_clear.append(i)
    words_text=Text(words_clear)
    
    return words_text

def vocab_stemmed(row_text=[]):
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    
    for i in row_text:
        allwords_stemmed = tokenize_and_stem(i)   
        totalvocab_stemmed.extend(allwords_stemmed)
        
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    #为简介中的所有词建立倒排索引
    inv_indx = {i:[] for i in totalvocab_stemmed}
    for word in totalvocab_stemmed:
        for i in range(len(row_text)):
            if word in row_text[i]:
                inv_indx[word].append(i)
    print(inv_indx)
    
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    
    for i in row_text:
        allwords_stemmed = tokenize_and_stem(i)   
        totalvocab_stemmed.extend(allwords_stemmed)
        
        allwords_tokenized = tokenize_only(i)
        totalvocab_tokenized.extend(allwords_tokenized)
    
    #print(words_counter)
    
    #词干化后的词和原词构成了一个对比词表
    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
    #print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
    return vocab_frame.head()

def tfidf(row_text=[]):
    #tfidf 向量原型
    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                     min_df=0.2, stop_words='english',
                                     use_idf=True, tokenizer=tokenize_only, 
                                     ngram_range=(1,3))
    
    #transform tfidf vector to tfidf matrix
    titleVectorizerArray = tfidf_vectorizer.fit_transform(
            row_text.values.astype('U')).toarray()
    #features of tf-idf
    terms = tfidf_vectorizer.get_feature_names()
    titledf = pd.DataFrame(titleVectorizerArray, columns= [terms])
    return titledf
