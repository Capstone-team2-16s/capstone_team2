import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from torch import optim


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from  nltk.stem import SnowballStemmer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from torch.utils.data.dataset import random_split
import time


# Model(수정 필요)
class CharCNN(nn.Module):
    def __init__(self):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(26, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc3 = nn.Linear(1024, 4)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.log_softmax(x)

        return x

#모델 구축
model = CharCNN()


# =============== 셋팅 =============== #

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

#전처리
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


train=pd.read_csv('C:/Users/VIP/Desktop/capstone/train.csv')
test=pd.read_csv('C:/Users/VIP/Desktop/capstone/test.csv')


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


t = Tokenizer()
t.fit_on_texts(train.text)

vocab_size = len(t.word_index) + 1



X_encoded = t.texts_to_sequences(train.text) #단어를 순차적인 숫자로 바꿔줍니다.

max_len=max(len(l) for l in X_encoded) #한 문장에서 최대 단어 개수를 반환
print(max_len)


X_train=pad_sequences(X_encoded, maxlen=max_len, padding='post')
#print(X_encoded)
#print(X_train)

# 모델의 state_dict 출력
#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#   print(param_tensor, "\t", model.state_dict()[param_tensor].size())



N_EPOCHS = 2
batch_size = 15
min_valid_loss = float('inf')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)






model.train()
#for epoch in range(0, N_EPOCHS):
#    running_loss = 0.0
#    for i in range(0, batch_size):
#        optimizer.zero_grad()
#        input = torch.as_tensor(X_train[i],device=device)
#        label = y_train[i]
#        print(input,label,"\n")
#        outputs = model(input)





#모델 저장
#torch.save(model.state_dict(),'C:/Users/VIP/Desktop/dataset')
