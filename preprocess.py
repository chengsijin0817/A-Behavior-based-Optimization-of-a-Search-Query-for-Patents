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
from nltk.tag import pos_tag

df = pd.read_csv('Alert.csv', encoding='latin-1', usecols=[0, 1])
ab = df['Abstract'][0]
print(ab)

#tokenize
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


