# 预训练的神经网络
from keras.applications import VGG16
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras import models
from keras import layers
from keras.applications.imagenet_utils import decode_predictions
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing import image


# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))               #预训练处理
# conv_base.summary()    #查看VGG16卷积层的情况特点

model =models.load_model("train_dir/cats_and_dogs_small_3.h5")
i = 0
val_dir = "datasets/cats_and_dogs_small/validation/cats"
img_paths = []
img_n = np.random.randint(1000, 1500, 10)
cat_dog = {0: "cat", 1: "dog"}
for i in img_n:
    p = os.path.join(val_dir, "cat.{}.jpg".format(i))
    img_paths.append(p)


img = []
# 把图片读取出来放到列表中
for i in range(len(img_paths)):
    images = image.load_img(img_paths[i], target_size=(150, 150))
    x = image.img_to_array(images)
    x = np.expand_dims(x, axis=0)
    img.append(x)

# 把图片数组联合在一起
x = np.concatenate([x for x in img])

y = list(map(lambda x: cat_dog[int(x)], model.predict(x)))
print(img_paths)

print(y)
print(model.predict(x))


