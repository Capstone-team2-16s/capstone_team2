#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd
from pandas import DataFrame as df

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nltk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Word2vec
import gensim
from gensim.models import Word2Vec #@
from gensim.utils import simple_preprocess #@
from gensim.models.keyedvectors import KeyedVectors #@

# Utility
import re
import numpy as np


# In[ ]:


# =============== 셋팅 =============== #

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

#전처리
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


# In[ ]:


#학습데이터 로드
train=pd.read_csv('./capstone/train.csv')
test=pd.read_csv('./capstone/test.csv')


# In[26]:


#train 데이터중 label 분리
y_train = train.sentiment
y_train = y_train.map({"negative": 0, "neutral": 2, "positive": 4})
y_train


# In[ ]:


#학습 데이터 텍스트 전처리
def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

train.text = train.text.apply(lambda x: preprocess(x))


# In[ ]:


train.text


# In[ ]:


t = Tokenizer()
t.fit_on_texts(train.text)


# In[ ]:


vocab_size = len(t.word_index) + 1
print(vocab_size) # 24414


# In[ ]:


X_encoded = t.texts_to_sequences(train.text) #단어를 순차적인 숫자로 바꿔줍니다.
#print(X_encoded)


# In[13]:


max_len=max(len(l) for l in X_encoded) #한 문장에서 최대 단어 개수를 반환
print(max_len) # 26


# In[14]:


X_train=pad_sequences(X_encoded, maxlen=max_len, padding='post') #벡터차원 고정을 위해 padding값 생성해줍니다.


# In[15]:


# =============== word2vec =============== #
# 구글의 사전 훈련된 Word2vec 모델을 로드
word2vec = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
print(word2vec.vectors.shape) # (3000000, 300)


# In[16]:


# 단어 집합 크기의 행과 300개의 열을 가지는 행렬 생성. 값은 전부 0
embedding_matrix = np.zeros((vocab_size, 300))
print(np.shape(embedding_matrix)) # (24414, 300)


# In[ ]:


word2vec.most_similar("love") #love와 비슷한 단어 찾기


# In[17]:


def get_vector(word):
    if word in word2vec:
        return word2vec[word]
    else:
        return None

for word, i in t.word_index.items(): # 훈련 데이터의 단어 집합에서 단어와 정수 인덱스를 1개씩 꺼내온다.
    temp = get_vector(word) # 단어(key) 해당되는 임베딩 벡터의 300개의 값(value)를 임시 변수에 저장
    if temp is not None: # 만약 None이 아니라면 임베딩 벡터의 값을 리턴받은 것이므로
        embedding_matrix[i] = temp # 해당 단어 위치의 행에 벡터의 값을 저장한다.


# In[27]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
 
model2 = Sequential()
model2.add(Embedding(vocab_size, 4, input_length=max_len)) # 모든 임베딩 벡터는 4차원.
model2.add(Flatten()) # Dense의 입력으로 넣기위함.
model2.add(Dense(1, input_shape=(1,)))
 
model2.compile('SGD','mse')
model2.fit(X_train, y_train, epochs=100, verbose=0)


# In[28]:


model2.summary()


# In[29]:


#트위터 샘플문장
samples=pd.read_csv('./capstone/sample.txt', sep = "\n", engine='python', encoding = "utf8",header=None)


# In[35]:


samples[0] = samples[0].apply(lambda x: preprocess(x))

y_encoded = t.texts_to_sequences(samples[0]) #단어를 순차적인 숫자로 바꿔줍니다.
y_test=pad_sequences(y_encoded, maxlen=26, padding='post')


# In[36]:


samples[0]
y_test


# In[39]:


#예측
y_predict=model2.predict(y_test)


# In[40]:


y_predict


# In[68]:


#다시 긍정중립
def sentiment(x):
    if x <1:
        return "negative"
    elif x <3:
        return "neutral"
    else:
        return "positive"

y=list(map(sentiment, y_predict))


# In[43]:





# In[69]:


#트위터 샘플문장
samples=pd.read_csv('./capstone/sample.txt', sep = "\n", engine='python', encoding = "utf8",header=None)


# In[70]:


#df1 = df(data={'감정':y_test,'문장':samples[0]})
df1 = df(samples[0])
df2 = df(y)
result3 = pd.concat([df1,df2],axis=1)
print(result3)


# In[ ]:




