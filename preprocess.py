#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:24:20 2018

@author: chengsijin
"""
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
patent_info = pd.read_excel(xlsx, 'Optimizer')


#import *csv file
#patent_info = pd.read_csv('Optimizer.csv', encoding='latin-1',na_values=['NA'])
"""
corpus = []
with open('Alert.csv', "r", encoding='latin-1') as f:
    for line in f.readlines():
        print(line)
        corpus.append(line.strip())
    #sent =  nltk.sent_tokenize(li)
print(corpus[1])
time.sleep(5)
"""
    
#    for i in nltk.sent_tokenize(f.read()):
#        words = nltk.word_tokenize(i)
#        print(words)


title = patent_info['Patent Title']
abstract = patent_info['Abstract']
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
    
    word_text=[]
    for a,b in nltk.pos_tag(tokens,tagset='universal'):
        if b=="ADJ" or "NOUN" or "VERB":
            word_text.append(a)
                            
    words_clear=[]
    for i in word_text:
        if i.lower() not in english_stopwords: # 过滤停用词
            if i not in english_punctuations: # 过滤标点符号
                if re.search('[a-z]', i):
                    words_clear.append(i)
    words_text=Text(words_clear)
    
    return words_text  

totalvocab_stemmed = []
totalvocab_tokenized = []

for i in range(185):
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


#tfidf 向量
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
 
tfidf_matrix = tfidf_vectorizer.fit_transform(patent_info['Abstract'].values.astype('U'))

print(tfidf_matrix)

#tf-idf 矩阵中的特征（features）表
terms = tfidf_vectorizer.get_feature_names()
print(terms)

dist = 1 - cosine_similarity(tfidf_matrix)
print(dist)


#clustering
from sklearn.cluster import KMeans
 
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print(clusters)

texts = []

# loop through document list  
for i in range(185):    
    allwords_stemmed = tokenize_and_stem(abstract[i])
    texts.append(allwords_stemmed) 
 
# turn our tokenized documents into a id <-> term dictionary  
dictionary = corpora.Dictionary(texts)  
      
# convert tokenized documents into a document-term matrix  
corpus = [dictionary.doc2bow(text) for text in texts]  
  
# generate LDA model  
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=1, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=1, num_words=10))


"""
    # transform documents to feature vectors
    count_vect = CountVectorizer()
    d_counts = count_vect.fit_transform(allwords_stemmed)
    #print(X_counts)
    X_counts.append(d_counts)
    
tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
X_tf = tf_transformer.transform(X_counts)
print("tf", X_tf)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
print("idf", X_tfidf)
    

# 定义向量化参数
count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(totalvocab_stemmed)

print(X_counts.shape)

tf_transformer = TfidfTransformer(use_idf=False).fit(X_counts)
X_tf = tf_transformer.transform(X_counts)
print("tf", X_tf)

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
print("idf", X_tfidf)



tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
 
tfidf_matrix = tfidf_vectorizer.fit_transform(totalvocab_stemmed)

print(tfidf_matrix.shape)


#print("词干化后的词", totalvocab_stemmed)
#print("仅分词", totalvocab_tokenized)
#vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
#print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')



#sentence = At eight o'clock on Thursday morning, Arthur didn't feel very good."
tokens = nltk.word_tokenize(ab)
#tagged = nltk.pos_tag(tokens)
#entities = nltk.chunk.ne_chunk(tagged)


# remove stopwords and punctuations
english_stopwords = stopwords.words("english")
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*'] # 自定义英文表单符号列表   

print("分词结果是：")
#Porter Stemming
st = PorterStemmer()  
    
words_clear=[]
for i in tokens:
    words_lower=i.lower()
    if i.lower() not in english_stopwords: # 过滤停用词
        if i not in english_punctuations: # 过滤标点符号
            t = st.stem(i) 
            words_clear.append(t)
print(words_clear)
#利用函数 Text() 将分词结果转换为 Text 格式，名称为 word_text            
word_text=Text(words_clear)   

#词频统计
words_counter=Counter(word_text)
print("词频统计",words_counter)

#留下形容词和名词
ADJ=[]
NOUN=[]
for a,b in pos_tag(word_text,tagset='universal'):
    if b=="ADJ":
        ADJ.append(a)
    elif b=="NOUN":
        NOUN.append(a)
print(len(ADJ))
print(len(NOUN))

c1=Counter(ADJ)
for i in c1.most_common(10):
    print(i[0],i[1])
c2=Counter(NOUN)
for i in c2.most_common(10):
    print(i[0],i[1])
"""
