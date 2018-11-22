from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import os
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

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())  # 将矩阵展开为（576，）
model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)#增强训练数据 shear_range 随意错切变换的角度
                                                                        # zoom_rang 图像随机缩放的范围fill_model 填充创新像素方法
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "datasets/cats_and_dogs_small/train"
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')#比较方便的构造器

validation_dir = "datasets/cats_and_dogs_small/validation"
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

#数据训练
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps =50)
model.save("./train_dir/cats_and_dogs_small_2.h5")




#图形作图

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')

plt.show()










