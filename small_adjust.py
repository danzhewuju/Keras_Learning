#神经网络的微调处理
from keras.applications import VGG16
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import models
from keras import layers
from keras.preprocessing import image
from keras import optimizers
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))      #将GPU设置为按需分配

base_datasets_dir = "datasets/train"

train_cats = "datasets/cats_and_dogs_small/train/cats"
train_dogs = "datasets/cats_and_dogs_small/train/dogs"

validation_cats = "datasets/cats_and_dogs_small/validation/cats"
validation_dogs = "datasets/cats_and_dogs_small/validation/dogs"

test_cats = "datasets/cats_and_dogs_small/test/cats"
test_dogs = "datasets/cats_and_dogs_small/test/dogs"


conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

# train_datagen = ImageDataGenerator(rescale= 1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
#                                    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "datasets/cats_and_dogs_small/train"
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

validation_dir = "datasets/cats_and_dogs_small/validation"
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')


# conv_base.summary()    #查看VGG16卷积层的情况特点
conv_base.trainble = True

set_trainble = False
for layer in conv_base.layers:
    if layer.name =='block5_conv1':
        layer.trainable = True
    else:
        layer.trainable = False


# model = models.Sequential()

model.compile(loss = 'binary_crossentroy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)



#图形作图

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')

plt.legend()                           #显示图例
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')

plt.legend()
plt.show()


