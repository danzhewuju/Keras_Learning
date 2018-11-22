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
from keras.callbacks import TensorBoard
from keras import layers
import os
from Unit import *
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))      #将GPU设置为按需分配




def learning_lstm():                   #lstm暂时还是比较适合于文本中，对于有序序暂不合适
    max_feature = 10000  #作为特征的单词个数
    maxlen = 500#设置最长的序列，如果长与这个序列会被阶段，短与这个序列会被填充
    # (x_train, y_labels), (x_test, y_test) = imdb.load_data(num_words=max_feature)
    # x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    # x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
    # # embedding_layer = Embedding(1000, 64)
    x_train, y_labels = data_test()
    x_train = np.expand_dims(x_train, axis=2)

    model = Sequential()
    # model.add(Embedding(max_feature, 32))
    model.add(LSTM(32, input_shape=(100, 1)))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    history = model.fit(x_train, y_labels, epochs=10, batch_size=128, validation_split=0.2)
    Draw(history)


def Lstm():
    max_feature = 2000  # 作为特征的单词个数
    maxlen = 500  # 设置最长的序列，如果长与这个序列会被阶段，短与这个序列会被填充
    (x_train, y_labels), (x_test, y_test) = imdb.load_data(num_words=max_feature)
    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)


    model = Sequential()
    model.add(Embedding(max_feature, 128, input_length=maxlen, name="embed"))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.MaxPooling1D(5))
    model.add(layers.Conv1D(32, 7, activation='relu'))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(1))
    model.summary()

    model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['acc'])
    callbacks =[
        TensorBoard(
            log_dir='my_log_dir', #日志文件的存储
            histogram_freq=1,  #每一轮后激活直方图
            embeddings_freq=1,#每一轮嵌入数据
            embeddings_layer_names="embed"
        )
    ]

    history = model.fit(x_train, y_labels, epochs=20, batch_size=128, validation_split=0.2, callbacks=callbacks)
    Draw(history)


def data_test():
    # x_train = np.ones((95, 100))
    x_train = np.random.randint(0, 2, (95, 100))
    # for index in range(95):
    #     for j in range(100):
    #         if index < j:
    #             x_train[index][j] = 0

    y_train = np.zeros(95)
    t1 = np.sum([np.sum(x) for x in x_train])
    t = np.sum(x_train)//95
    for i in range(95):
        if sum(x_train[i]) >= t:
            y_train[i] = 1
    return x_train, y_train


Lstm()
# print(data_test())

# print(test())