from Unit import *
from keras.datasets import imdb
from keras.models import Sequential
from keras import layers
from keras.preprocessing import sequence
from  keras.optimizers import RMSprop
max_feature = 10000
max_len = 500
print("Loading data......")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)

x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)


model = Sequential()
model.add(layers.Embedding(max_feature, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))

model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc']
              )
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
Draw(history)




