import numpy as np
#import codecs
import matplotlib.pyplot as plt
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
import re
import spacy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import torch
import torch.nn as nn



input_file = open("C:/kaggle/test.csv", "r",encoding='utf-8', errors='replace')
train = pd.read_csv(input_file)
#print(train.text)

list_labels = train.sentiment
list_labels =list_labels.map({"negative": 0, "neutral": 1, "positive": 2})
#print(list_labels)

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

# clean data
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
nlp = spacy.load("en_core_web_sm")
doc = train["text"].apply(nlp)
print(doc)


max_sent_len=max(len(doc[i]) for i in range(0,len(doc)))
print("length of longest sentence: ", max_sent_len)
vector_len=len(doc[0][0].vector)
print("length of each word vector: ", vector_len)


#creating the 3D array
tweet_matrix=np.zeros((len(doc),max_sent_len,vector_len))
print(tweet_matrix[0:2,0:3,0:4]) #test print

for i in range(0,len(doc)):
    for j in range(0,len(doc[i])):
        tweet_matrix[i][j]=doc[i][j].vector

#create label

print(list_labels.shape[0])
print(tweet_matrix.shape[0])

len_for_split=[int(tweet_matrix.shape[0]/4),int(tweet_matrix.shape[0]*(3/4))]
print(len_for_split)
len_for_split[0]+=1
test, train=random_split(tweet_matrix,len_for_split)
print(test.dataset.shape)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_epochs = 25
num_classes = 3
learning_rate = 0.001
batch_size=100


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)
#load labels #truncating total data to keep batch size 100
labels_train=list_labels[train.indices[0:2600]]
labels_test=list_labels[test.indices[0:880]]

#load train data
training_data=train.dataset[train.indices[0:2600]].astype(float)

#training_data=training_data.unsqueeze(1)

#load test data
test_data=test.dataset[test.indices[0:880]].astype(float)
#test_data=test_data.unsqueeze(1)

dataset_train = MyDataset(training_data, labels_train)
dataset_test = MyDataset(test_data, labels_test)


#loading data batchwise
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer13 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(3, vector_len), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(70, 1), stride=1))
        self.layer14 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(4, vector_len), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(69, 1), stride=1))
        self.layer15 = nn.Sequential(
            nn.Conv2d(1, 100, kernel_size=(5, vector_len), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(68, 1), stride=1))
        # self.layer2 = nn.Sequential(
        # nn.Conv2d(15, 30, kernel_size=5, stride=1, padding=0),
        # nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        # concat operation
        self.fc1 = nn.Linear(1 * 1 * 100 * 3, 30)
        self.fc2 = nn.Linear(30, 3)
        # self.fc3 = nn.Linear(100,3)

    def forward(self, x):
        x3 = self.layer13(x)
        x4 = self.layer14(x)
        x5 = self.layer15(x)
        x3 = x3.reshape(x3.size(0), -1)
        x4 = x4.reshape(x4.size(0), -1)
        x5 = x5.reshape(x5.size(0), -1)
        x3 = self.drop_out(x3)
        x4 = self.drop_out(x4)
        x5 = self.drop_out(x5)
        out = torch.cat((x3, x4, x5), 1)
        out = self.fc1(out)
        out = self.fc2(out)
        return (out)

#creating instance of our ConvNet class
model = ConvNet()
model.to(device) #CNN to GPU


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
#CrossEntropyLoss function combines both a SoftMax activation and a cross entropy loss function in the same function

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = 8100 / batch_size

loss_list = []
acc_list = []
val_acc_list = []

for epoch in range(num_epochs):
    loss_list_element = 0
    acc_list_element = 0
    for i, (data_t, labels) in enumerate(train_loader):
        data_t = data_t.unsqueeze(1)
        data_t, labels = data_t.to(device), labels.to(device)

        # Run the forward pass
        outputs = model(data_t)
        loss = criterion(outputs, labels)
        loss_list_element += loss.item()
        # print("==========forward pass finished==========")

        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print("==========backward pass finished==========")

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        acc_list_element += correct

    loss_list_element = loss_list_element / np.shape(labels_train)[0]
    acc_list_element = acc_list_element / np.shape(labels_train)[0]
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, loss_list_element, acc_list_element * 100))
    loss_list.append(loss_list_element)
    acc_list.append(acc_list_element)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data_t, labels in test_loader:
        data_t = data_t.unsqueeze(1)
        data_t, labels = data_t.to(device), labels.to(device)

        outputs = model(data_t)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model: {} %'.format((correct / total) * 100))
