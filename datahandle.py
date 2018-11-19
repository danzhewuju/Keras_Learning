import os, shutil
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image

base_datasets_dir = "datasets/train"

train_cats = "datasets/cats_and_dogs_small/train/cats"
train_dogs = "datasets/cats_and_dogs_small/train/dogs"

validation_cats = "datasets/cats_and_dogs_small/validation/cats"
validation_dogs = "datasets/cats_and_dogs_small/validation/dogs"

test_cats = "datasets/cats_and_dogs_small/test/cats"
test_dogs = "datasets/cats_and_dogs_small/test/dogs"

# fnames = ["cat.{}.jpg".format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(base_datasets_dir, fname)
#     dst = os.path.join(train_cats, fname)
#     shutil.copyfile(src, dst)

# fnames = ["cat.{}.jpg".format(i) for i in range(1000, 1500)]
# for fname in fnames:
#     src = os.path.join(base_datasets_dir, fname)
#     dst = os.path.join(validation_cats, fname)
#     shutil.copyfile(src, dst)

# fnames = ["cat.{}.jpg".format(i) for i in range(1500, 2000)]
# for fname in fnames:
#     src = os.path.join(base_datasets_dir, fname)
#     dst = os.path.join(test_cats, fname)
#     shutil.copyfile(src, dst)

# fnames = ["dog.{}.jpg".format(i) for i in range(1000)]
# for fname in fnames:
#     src = os.path.join(base_datasets_dir, fname)
#     dst = os.path.join(train_dogs, fname)
#     shutil.copyfile(src, dst)

# fnames = ["dog.{}.jpg".format(i) for i in range(1000,1500)]
# for fname in fnames:
#     src = os.path.join(base_datasets_dir, fname)
#     dst = os.path.join(validation_dogs, fname)
#     shutil.copyfile(src, dst)

datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')


fnames = [os.path.join(train_cats, fname) for fname in os.listdir(train_cats)]
img_path = fnames[20]
img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size = 1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 10 == 0:
        break
plt.show()