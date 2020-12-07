# =============== Import Library =============== #
import pandas as pd
from pandas import DataFrame as df

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# nltk
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

#Keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model #모델 저장

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Utility
import re
import numpy as np
import time #수행시간 측정
from collections import Counter

# emoji
import emoji
import json

# =============== Setting Value =============== #

# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8
MAX_LEN = 50
VOCAB_SIZE = 400000

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

#전처리
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

####
###
##
#

#Learned Tokenizer import
tk = Tokenizer(num_words=VOCAB_SIZE)
with open('../modelData/wordIndex.json') as json_file:
  word_index = json.load(json_file)
  tk.word_index = word_index

#Load Model
# model = load_model('../modelData/pruned_tCNN.h5')
model = load_model('../modelData/pruned80_tCNN.h5')
#decode function
def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE

        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE
#predict function
def predict(ex_text, include_neutral=True):
    start_at = time.time()
    x_encoded = tk.texts_to_sequences([ex_text])
    res_test=np.array(pad_sequences(x_encoded, maxlen=MAX_LEN, padding='post'))
    # Predict
    score = model.predict([res_test])
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)
    
    return {"label": label, "score": float(score),
       "elapsed_time": time.time()-start_at}


# =============== predict sentiment with Twits =============== #

class preproc_Sentence:
    def __init__(self):
        pass

    # def readTweets(request_id):
    #     id = request_id
    #     file_name = 'twitter_'
    #     fileformat = '.txt'
    #     filename = file_name + id + fileformat

    #     data_path ='../data/'

    #     # 분석 요청된 유명인 트윗 파일 open
    #     with open(data_path + filename, 'r', encoding = "utf-8") as f:
    #         tweets = pd.read_csv(f, sep = "\n", names = ['data'])
    #     f.close()

    #     return tweets

    def preprocTweets(tweets):        
        # URL 변환
        tweets['data'] = tweets['data'].replace(to_replace = "((www\.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+))", value = "URL ", regex = True)
        # 소문자 변환
        tweets['preprocess'] = tweets['data'].str.lower()
        # @ 변환
        tweets['preprocess'] = tweets['preprocess'].replace(to_replace = "'@[^\s]+", value = "USERID", regex = True)
        # hashtag 변환
        tweets['preprocess'] = tweets['preprocess'].replace(to_replace = "#([^\s]+)", value = "HASHTAG", regex = True)
        # hashtag 변환
        tweets['preprocess'] = tweets['preprocess'].replace(to_replace = "([a-zA-Z0-9_.+-]@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)", value = "EMAIL", regex = True)
        # Emoji 변환
        tweets_raw = tweets['preprocess']

        for i in range(len(tweets_raw)):
            tweets_raw[i] = emoji.demojize(tweets_raw[i], use_aliases = True)

        tweets['preprocess'] = tweets_raw

        return tweets

class preproc_Word:
    def __init__(self):
        pass

    # def readTweet(request_id):
    #     id = request_id
    #     file_name = 'twitter_'
    #     fileformat = '.txt'
    #     filename = file_name + id + fileformat

    #     data_path = '../data/'

    #     # 분석 요청된 유명인 트윗 파일 open
    #     with open(data_path + filename, 'r', encoding = "utf-8") as file:
    #         tweet = file.read()
       
    #     return tweet

    def preprocWordTweet(tweets):
        #dataframe to string
        tweet = str(tweets['data'])
        # 소문자 변환
        tweet = tweet.lower()
        # URL 제거
        tweet = re.sub('((www\.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+))', '', tweet)
        # 구두점 제거
        tweet = re.sub(r'[^\w\s]', '', tweet)
        # 숫자 제거
        tweet = re.sub('\s[0-9]+', '', tweet)
        # 아이디 제거
        tweet = re.sub('@[A-Za-z0-9]+', '', tweet)
        # 이메일 제거
        tweet = re.sub('([a-zA-Z0-9_.+-]@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+)', '', tweet)

        return tweet
    
    def tokenizeWord(tweet):
        stop_words = set(stopwords.words('english')) 
        word_tokens = word_tokenize(tweet)
 
        res = []
        for w in word_tokens: 
            if w not in stop_words: 
                res.append(w)
        return res
    
    def stemmerWord(res):
        stemmer = SnowballStemmer('english')
        words = [stemmer.stem(w) for w in res]
        return words

class word_COUNT:
    def __init__(self):
        pass

    def countWord(tweets):
        tweet = preproc_Word.preprocWordTweet(tweets)
        res = preproc_Word.tokenizeWord(tweet)
        words = preproc_Word.stemmerWord(res)
        words_top = Counter(words).most_common(n=5)
        wordList =[]
        wordCountList = [] 
        for i in range(len(words_top)):
            wordList.append(words_top[i][0]) 
            wordCountList.append(words_top[i][1])
        tmpDict={}
        resDict={}
        tmpDict["words"]=wordList
        tmpDict["counts"]=wordCountList
        resDict["countWords"]=tmpDict
        
        return resDict

class tweet_SentimentAnalyse :
    def __init__(self):
        pass

    def sentimentAnalyse(tweets_data) :
        # 결과 dataframe 생성
        df_res = pd.DataFrame({'text':[], 'label':[], 'score':[], 'elapsed_time':[]})
        #print('트윗 문장 감정 분석 결과')
        for col,item in tweets_data.iterrows():
            # predict class로 수정 필요
            res = predict(item[1])
            df_res.loc[col] = [item[0], res['label'], res['score'],res['elapsed_time'] ]
        #print(df_res)
        #print()
        return df_res

    def countTypes(df_res):
        #트윗 문장 감정비율
        res = df_res['label'].value_counts(normalize=True).mul(100).round(2).astype(str)+'%'
        tmp1={}
        tmp1["label"]=["POSITIVE","NEGATIVE","NEUTRAL"]
        tmp1["ratio"]=[res["POSITIVE"],res["NEGATIVE"],res["NEUTRAL"]]
        df_pos = df_res.sort_values(by="score", ascending=False).head(2)
        df_neg = df_res.sort_values(by="score", ascending=True).head(2)
        df_res['cal'] = abs(df_res['score'] - 0.5)
        df_neu = df_res.sort_values(by="cal", ascending=True).head(2)
        
        list_pos = list(df_pos['text'])
        list_neg = list(df_neg['text'])
        list_neu = list(df_neu['text'])

        tmp2={}
        tmp2["pos"]=list_pos
        tmp2["neg"]=list_neg
        tmp2["neu"]=list_neu
        
        tmp1["sentence"]=tmp2

        resDict={}
        resDict["sentiment"]=tmp1
        return resDict


