#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:23:44 2019

@author: JeremyKulchyk
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Masking
from keras.layers import Dropout
from keras.layers import SimpleRNN

ticker_symbol = "FB"

df = pd.read_csv("880DataSet_WithClean.csv",encoding = "ISO-8859-1")

df1 = df[(df["symbol"] == ticker_symbol) & (df["tri_change"] != "2")]

# Turn each day into a feature vector.
Dates = sorted(list(set(list(df1["date"]))))

#Find max number of tweets in one day
M = 0
for x in range(0,len(Dates)):
    df2 = df1[df1["date"] == Dates[x]]
    if len(df2) > M:
        M = len(df2)
    else:
        continue


cols = ['symbol','date','friends_count', 'followers_count','favourites_count','volume', 'sent','bin_change','tri_change']
#subcols = ['friends_count', 'followers_count','favourites_count', 'sent']
HStates = 5
TotLen = len(Dates)
#Num_Feats = len(subcols)
Window = M

df = df1[cols]
#df[subcols] = (df[subcols] - df[subcols].min()) / (df[subcols].max() - df[subcols].min())


df1Seg = [] #x values
dfySeg = [] # corresponding output values for each above segment

for i in range(0, len(Dates)):  # For each date
    df2 = df[(df["date"] == Dates[i]) & (df["symbol"] == ticker_symbol)]
    dfySeg.append(int(df2["tri_change"].iloc[0]))

    # Avg
    Avg_friends_count = np.mean(df2['friends_count'].iloc[0:len(df2)].values)
    Avg_followers_count = np.mean(df2['followers_count'].iloc[0:len(df2)].values)
    Avg_favourites_count = np.mean(df2['favourites_count'].iloc[0:len(df2)].values)
    Avg_sent = np.mean(df2['sent'].iloc[0:len(df2)].values)
    
    # Min
    min_friends_count = np.min(df2['friends_count'].iloc[0:len(df2)].values)
    min_followers_count = np.min(df2['followers_count'].iloc[0:len(df2)].values)
    min_favourites_count = np.min(df2['favourites_count'].iloc[0:len(df2)].values)
    min_sent = np.min(df2['sent'].iloc[0:len(df2)].values)
    
    # Max
    max_friends_count = np.max(df2['friends_count'].iloc[0:len(df2)].values)
    max_followers_count = np.max(df2['followers_count'].iloc[0:len(df2)].values)
    max_favourites_count = np.max(df2['favourites_count'].iloc[0:len(df2)].values)
    max_sent = np.max(df2['sent'].iloc[0:len(df2)].values)
    
    # STD
    std_friends_count = np.std(df2['friends_count'].iloc[0:len(df2)].values)
    std_followers_count = np.std(df2['followers_count'].iloc[0:len(df2)].values)
    std_favourites_count = np.std(df2['favourites_count'].iloc[0:len(df2)].values)
    std_sent = np.std(df2['sent'].iloc[0:len(df2)].values)
    
    # PTP
    ptp_friends_count = np.ptp(df2['friends_count'].iloc[0:len(df2)].values)
    ptp_followers_count = np.ptp(df2['followers_count'].iloc[0:len(df2)].values)
    ptp_favourites_count = np.ptp(df2['favourites_count'].iloc[0:len(df2)].values)
    ptp_sent = np.ptp(df2['sent'].iloc[0:len(df2)].values)
    
    # SUM
    sum_friends_count = np.sum(df2['friends_count'].iloc[0:len(df2)].values)
    sum_followers_count = np.sum(df2['followers_count'].iloc[0:len(df2)].values)
    sum_favourites_count = np.sum(df2['favourites_count'].iloc[0:len(df2)].values)
    sum_sent = np.sum(df2['sent'].iloc[0:len(df2)].values)

    # CUMSUM
    """
    cumsum_friends_count = np.cumsum(df2['friends_count'].iloc[0:len(df2)].values)
    cumsum_followers_count = np.cumsum(df2['followers_count'].iloc[0:len(df2)].values)
    cumsum_favourites_count = np.cumsum(df2['favourites_count'].iloc[0:len(df2)].values)
    cumsum_sent = np.cumsum(df2['sent'].iloc[0:len(df2)].values)
    
    # CUMPROD
    cumprod_friends_count = np.cumprod(df2['friends_count'].iloc[0:len(df2)].values)
    cumprod_followers_count = np.cumprod(df2['followers_count'].iloc[0:len(df2)].values)
    cumprod_favourites_count = np.cumprod(df2['favourites_count'].iloc[0:len(df2)].values)
    cumprod_sent = np.cumprod(df2['sent'].iloc[0:len(df2)].values)
    """
    Vol = df2['volume'].iloc[0]
    #Change = df2['tri_change'].iloc[0]

    FeatArr = [Avg_friends_count,Avg_followers_count,Avg_favourites_count,Avg_sent,
               min_friends_count,min_followers_count,min_favourites_count,min_sent,
               max_friends_count,max_followers_count,max_favourites_count,max_sent,
               std_friends_count,std_followers_count,std_favourites_count,std_sent,
               ptp_friends_count,ptp_followers_count,ptp_favourites_count,ptp_sent,
               sum_friends_count,sum_followers_count,sum_favourites_count,sum_sent,Vol]
    
    FeatAr = np.array(FeatArr)

    df1Seg.append(FeatAr)

train_X = np.zeros((TotLen, len(FeatAr)))  # 1 row per day, each row is feature vector
train_Y = np.zeros(TotLen)

for j in range(0,len(df1Seg)):
    train_X[j] = df1Seg[j]
    train_Y[j] = int(dfySeg[j])

train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))  # Make shape correct for keras to be happy

X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=.3, random_state=0)

# hyperparameters
epochs = 10
batch_size = 5
layer1Size = 20
layer2Size = 20
layer3Size = 10
layer4Size = 10
layer5Size = 5

# dataset size
inpLen = X_train.shape[1]
inpDim = X_train.shape[2]
print(X_train.shape)

# Create model
model = Sequential()

# # CNN Model
# model.add(Dense(layer1Size, input_dim=inpDim, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(layer2Size, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(layer3Size, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(layer4Size, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(layer5Size, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# RNN Model
model.add(SimpleRNN(32, activation='relu', input_shape=(inpLen, inpDim)))
model.add(Dense(1, activation='sigmoid'))

# # LSTM Model - Doesnt work
# model.add(Masking(mask_value=0.0))
# model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
# model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
print(model.summary())


# Evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Training Finished ")
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


"""

model=Sequential()
model.add(SimpleRNN(2, activation='relu', input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dense(2))
model.compile(optimizer=keras.optimizers.Adam(),metrics=['accuracy'], loss=keras.losses.sparse_categorical_crossentropy)
print(model.summary())
model.fit(X_train, y_train, epochs=10, batch_size=5,verbose=1)
scores = model.evaluate(X_test, y_test, verbose=1)

y_pred = model.predict(X_test)
y_test1 = y_test
y_test = label_binarize(y_test, classes=[x for x in range(0,2)])

y_pred = model.predict(X_test)
ypred = []
for List in y_pred:
    M = max(List)
    i = np.where(List == M)[0][0]
    ypred.append(int(i))
y_pred = ypred

cm = confusion_matrix(y_test1, y_pred)
Accuracy = accuracy_score(y_test1, y_pred)
#F1 = f1_score(y_test1, y_pred, average="macro")
#Precision = precision_score(y_test1, y_pred, average="macro")
Recall = recall_score(y_test1, y_pred, average="macro")
print(Accuracy)
print(cm)


train_X = train_X.reshape(-1, Num_Feats,Window, 1)

batch_size = 64
num_classes = 3


X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=.3, random_state=0)

CNN1 = Sequential()
#CNN1.add(keras.layers.Masking(mask_value=0.0, input_shape=(Num_Feats,Window,1)))
CNN1.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(Num_Feats,Window,1),padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))
CNN1.add(MaxPooling2D((2, 2),padding='same'))
CNN1.add(Dropout(0.25))
CNN1.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))
CNN1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN1.add(Dropout(0.25))
CNN1.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
CNN1.add(LeakyReLU(alpha=0.1))                  
CNN1.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
CNN1.add(Dropout(0.25))
CNN1.add(Flatten())
CNN1.add(Dense(128, activation='linear'))
CNN1.add(LeakyReLU(alpha=0.1))                  
CNN1.add(Dropout(0.25))
CNN1.add(Dense(num_classes, activation='softmax'))
CNN1.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
print(CNN1.summary())
CNN1.fit(X_train, y_train, batch_size=batch_size,epochs=10,verbose=1)
scores = CNN1.evaluate(X_test, y_test, verbose=1)

y_pred = CNN1.predict(X_test)
y_test1 = y_test
y_test = label_binarize(y_test, classes=[x for x in range(0,num_classes)])

y_pred = CNN1.predict(X_test)
ypred = []
for List in y_pred:
    M = max(List)
    i = np.where(List == M)[0][0]
    ypred.append(int(i))
y_pred = ypred

cm = confusion_matrix(y_test1, y_pred)
Accuracy = accuracy_score(y_test1, y_pred)
#F1 = f1_score(y_test1, y_pred, average="macro")
#Precision = precision_score(y_test1, y_pred, average="macro")
Recall = recall_score(y_test1, y_pred, average="macro")
print(Accuracy)
print(cm)
"""


