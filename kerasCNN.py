import keras
import numpy as np # linear algebra
import pandas as pd
from keras.preprocessing import sequence
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pandas import DataFrame as df


TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

#전처리
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

train=pd.read_csv('C:/kaggle/train.csv')
test=pd.read_csv('C:/kaggle/test.csv')


y_train = train.sentiment
y_train = y_train.map({"negative": 0, "neutral": 2, "positive": 4})


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
print(train.text)
tweets = np.array(train.text)
sentiment = np.array(y_train)
print(sentiment)



vocab_size = 400000
tk = Tokenizer(num_words=vocab_size)
#tw = tweets
tk.fit_on_texts(tweets)
t = tk.texts_to_sequences(tweets)
X = np.array(sequence.pad_sequences(t, maxlen=26, padding='post'))
y = sentiment

print(X.shape, y.shape)


#Make Model
model = Sequential()

model.add(Embedding(vocab_size, 32, input_length=26))
model.add(Conv1D(filters=128, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=64, kernel_size=6, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=7, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=8, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, input_shape=(1,)))
model.compile('SGD','mse',metrics=['accuracy'])
model.summary()

model.fit(X, y, epochs=10, verbose=1)
model.save('model.h5')




#prediction
samples=pd.read_csv('C:/kaggle/sample.txt', sep = "\n", engine='python', encoding = "utf8",header=None)
samples[0] = samples[0].apply(lambda x: preprocess(x))

y_encoded = tk.texts_to_sequences(samples[0]) #단어를 순차적인 숫자로 바꿔줍니다.
y_test=pad_sequences(y_encoded, maxlen=26, padding='post')

y_predict=model.predict(y_test)
def sentiment(x):
    if x <1:
        return "negative"
    elif x <3:
        return "neutral"
    else:
        return "positive"

y=list(map(sentiment, y_predict))
samples=pd.read_csv('C:/kaggle/sample.txt', sep = "\n", engine='python', encoding = "utf8",header=None)


# In[70]:


#df1 = df(data={'감정':y_test,'문장':samples[0]})
df1 = df(samples[0])
df2 = df(y)
result3 = pd.concat([df1,df2],axis=1)
print(result3)
