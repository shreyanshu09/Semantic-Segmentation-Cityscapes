import numpy as np
import os
import matplotlib.pyplot as plot
from PIL import Image
import cv2
import random
import seaborn as sns
import tensorflow as tf
from math import ceil
import time

train_folder = "cityscapes_data/train"
valid_folder = "cityscapes_data/val"
width = 256
height = 256
classes = 13
batch_size = 10
num_of_training_samples = len(os.listdir(train_folder))
num_of_testing_samples = len(os.listdir(valid_folder))
print(num_of_training_samples)
print(num_of_testing_samples)


def LoadImage(name, path):
    img = Image.open(os.path.join(path, name))
    img = np.array(img)

    image = img[:, :256]
    mask = img[:, 256:]

    return image, mask


def getSegmentationArr(mask, classes, width=width, height=height):
    masked = mask.reshape((mask.shape[0] * mask.shape[1], 3))
    pred = kmeans.predict(masked)
    pred.shape
    pred = pred.reshape(mask.shape[0], mask.shape[1])

    seg_labels = np.zeros((height, width, classes))

    for c in range(classes):
        seg_labels[:, :, c] = (pred == c).astype(int)
    return seg_labels


train_list = os.listdir('cityscapes_data/train')
from tqdm import tqdm

colors = []

for i in tqdm(range(150)):
    x, y = LoadImage(train_list[i], train_folder)
    colors.append(y.reshape(y.shape[0] * y.shape[1], 3))

colors = np.array(colors)
colors = colors.reshape((colors.shape[0] * colors.shape[1], 3))
colors.shape

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=13, random_state=0)
kmeans.fit(colors)

kmeans.cluster_centers_

l = list(kmeans.labels_)
s = set(l)
s


def give_color_to_seg_img(seg, n_classes=classes):
    seg_img = np.zeros((seg.shape[0], seg.shape[1], 3)).astype('float')
    colors = sns.color_palette("hls", n_classes)

    for c in range(n_classes):
        segc = (seg == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    return (seg_img)


def addColors(gg):
    im = np.zeros([gg.shape[0], gg.shape[1], 3], dtype=np.uint8)
    for i in range(gg.shape[0]):
        for u in range(gg.shape[1]):
            if gg[i, u] == 0:
                im[i, u] = np.array([7.97324365, 100.09284559,
                                     3.3311774])  # np.array([106.90082868, 139.99479298,  36.44992868]) #dirty-green
            if gg[i, u] == 12:
                im[i, u] = np.array([125.15370551, 128.00683271, 102.70661342])
            if gg[i, u] == 11:
                im[i, u] = np.array([205.10936684, 155.91383531, 158.10853995])
            if gg[i, u] == 10:
                im[i, u] = np.array([202.89782929, 26.40039899, 61.60446492])
            if gg[i, u] == 9:
                im[i, u] = np.array([69.55103943, 70.40548991, 69.17557542])  # road
            if gg[i, u] == 8:
                im[i, u] = np.array([127.97324365, 63.09284559, 127.3311774])

            if gg[i, u] == 7:
                im[i, u] = np.array([76.50791694, 126.13882776, 172.87875815])  # sky
            if gg[i, u] == 6:
                im[i, u] = np.array([157.75659272, 245.35283586, 155.30654771])  # road-dividers
            if gg[i, u] == 4:
                im[i, u] = np.array([80.53963208, 6.04446257, 71.14193837])  # buildings
            if gg[i, u] == 5:
                im[i, u] = np.array([3.55582649, 3.56494346, 136.37082893])  # vehicles
            if gg[i, u] == 3:
                im[i, u] = np.array([237.59908029, 39.26874128, 225.79570494])
            if gg[i, u] == 2:
                im[i, u] = np.array([4.1605802, 3.27185434, 6.7030066])
            if gg[i, u] == 1:
                im[i, u] = np.array([214.7472683, 206.44713466, 33.15308545])
    return im


im, mask = LoadImage(train_list[4], train_folder)
c1 = getSegmentationArr(mask, classes, width=width, height=height)
c = addColors(np.argmax(c1, axis=-1))
fig, ax = plot.subplots(1, 2, figsize=(10, 10))
ax[0].imshow(mask)
ax[1].imshow(c)
plot.show()


def DataGenerator(path, batch_size=10, classes=13):
    files = os.listdir(path)
    while True:
        for i in range(0, len(files), batch_size):
            batch_files = files[i: i + batch_size]
            imgs = []
            segs = []
            for file in batch_files:
                image, mask = LoadImage(file, path)
                labels = getSegmentationArr(mask, classes)

                imgs.append(image)
                segs.append(labels)

            yield np.array(imgs), np.array(segs)


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomResizedCrop(255, scale=(0.5,0.9), ratio=(1, 1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.cityscapes_train(root='train_folder', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.cityscapes_val(root='valid_folder', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

train_gen = DataGenerator(trainloader, batch_size=batch_size)
val_gen = DataGenerator(testloader, batch_size=batch_size)
imgs, segs = next(train_gen)
imgs.shape, segs.shape

image = imgs[7]
mask = addColors(np.argmax(segs[7], axis=-1))
masked_image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

fig, axs = plot.subplots(1, 3, figsize=(20, 20))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[1].imshow(mask)
axs[1].set_title('Segmentation Mask')
axs[2].imshow(masked_image)
axs[2].set_title('Masked Image')
plot.show()

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


def Unet(n_filters=16, dilation_rate=1, n_classes=32):
    
    # Define input batch shape
    inputs = Input(shape=(256, 256, 3))

    conv1 = Conv2D(n_filters * 1, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(inputs)
    conv1 = Conv2D(n_filters * 1, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv1)

    conv2 = Conv2D(n_filters * 2, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(pool1)
    conv2 = Conv2D(n_filters * 2, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv2)

    conv3 = Conv2D(n_filters * 4, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(pool2)
    conv3 = Conv2D(n_filters * 4, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv3)

    conv4 = Conv2D(n_filters * 8, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(pool3)
    conv4 = Conv2D(n_filters * 8, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(conv4)

    conv5 = Conv2D(n_filters * 16, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(pool4)
    conv5 = Conv2D(n_filters * 16, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv5)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

    conv6 = Conv2D(n_filters * 8, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(up6)
    conv6 = Conv2D(n_filters * 8, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv6)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

    conv7 = Conv2D(n_filters * 4, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(up7)
    conv7 = Conv2D(n_filters * 4, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv7)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

    conv8 = Conv2D(n_filters * 2, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(up8)
    conv8 = Conv2D(n_filters * 2, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv8)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

    conv9 = Conv2D(n_filters * 1, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(up9)
    conv9 = Conv2D(n_filters * 1, (3, 3), activation='elu', padding='same', dilation_rate=dilation_rate)(conv9)
    conv10 = Conv2D(n_classes, (1, 1), activation='softmax', padding='same', dilation_rate=dilation_rate)(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


model = Unet(n_filters=32, n_classes=classes)
opt = tf.keras.optimizers.SGD(lr=0.0001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()

history = model.fit(train_gen, epochs=10, steps_per_epoch=num_of_training_samples // batch_size,
                    validation_data=val_gen, validation_steps=num_of_testing_samples // batch_size)

import cv2

m = '205.jpg'
m, tru = LoadImage(m, valid_folder)
i = np.asarray(m)
i = i.reshape((1, 256, 256, 3))
i.shape

r = model.predict(i)
r = r.reshape((256, 256, -1))
s = addColors(np.argmax(r, axis=-1))
plot.imshow(s)

from cv2 import *

fig, axs = plot.subplots(1, 4, figsize=(20, 20))
axs[0].imshow(m)
axs[0].set_title('Original Image')
axs[1].imshow(tru)
axs[1].set_title('True Mask')
axs[2].imshow(s)
axs[2].set_title('Predicted')
masked_image = cv2.addWeighted(m, 0.4, s, 0.6, 0)
axs[3].imshow(masked_image)
plot.show()

from cv2 import *

fig, axs = plot.subplots(1, 3, figsize=(20, 20))
axs[0].imshow(m)
axs[0].set_title('Original Image')
axs[1].imshow(s)
axs[1].set_title('Predicted')
masked_image = cv2.addWeighted(m, 0.5, s, 0.9, 0)
axs[2].imshow(masked_image)
axs[2].set_title('Masked Image')
plot.show()