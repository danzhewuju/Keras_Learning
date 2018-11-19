from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras.preprocessing as preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding
import numpy as np
import os
from Unit import *


def learning_lstm():
    max_feature = 10000
    maxlen = 64
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    # embedding_layer = Embedding(1000, 64)

    model = Sequential()
    model.add(Embedding(max_feature, 32))
    model.add(LSTM(32))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
    Draw(history)