import keras
import numpy as np
import collections
from keras import Sequential
from  keras import layers
import os, sys
import random

path = keras.utils.get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(path).read().lower()
# f = open("./datasets/nietzsche.txt", 'w', encoding="UTF-8")
# f.write(text)
# f.close()
print('Corpus length', len(text))

maxlen = 60
step = 3
sentences = []
next_chars = []

for i in range(0, len(text)-maxlen, step):
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

print('Number of Sequence:', len(sentences))
chars_r1 = collections.Counter(text)
chars_r1 = sorted(chars_r1.items(), key=lambda x: -x[1])
words, _ = zip(*chars_r1)
word_num_map = dict(zip(words, range(len(chars_r1))))
chars = sorted(list(set(text)))                #生成唯一字符的列表
print('Unique characters:', len(chars))
chars_indices = dict([(char, chars.index(char)) for char in chars])  #生成一个字典将唯一的字符进行映射

print('Vectirization...')

#将字符one-hot编码为二进制数组
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, chars_indices[char]] = 1
    y[i, chars_indices[next_chars[i]]] = 1

#预测下一个字符的单层lstm模型

model = Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, tempreature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / tempreature
    exp_preds = np.exp(preds)
    preds = exp_preds/np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


#文本生成序列
for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)#模型在数据上进行拟合一次
    start_index = random.randint(0, len(text)-maxlen-1)
    generated_text = text[start_index: start_index+maxlen]
    print('---Generationg with seed: %s ' % generated_text)
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print("----------temperature:{}\n".format(temperature))  #调整不同过的采样温度
        sys.stdout.write(generated_text)

        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, chars_indices[char]] = 1
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]

            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)


