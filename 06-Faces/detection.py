from skimage.io import imread, imshow
import numpy as np
import pandas as pd
import os
from skimage.transform import resize
from skimage import img_as_float
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import keras

def normalize(img):
    img -= img.mean()
    img /= np.sqrt((img**2).mean())

def prepare_img(image, tg_size):
    img = resize(image, (tg_size, tg_size))
    if len(img.shape) == 2:
        normalize(img)
        return np.array([img, img, img]).transpose((1,2,0))
    else:
        for i in range(img.shape[2]):
            normalize(img[:,:,i])
        return img

def get_data_shapes_filenames(directory, tg_size=128):
    filenames = sorted(os.listdir(directory))
    result = np.zeros((len(filenames), tg_size, tg_size, 3))
    shapes = np.zeros((len(filenames), 2))
    for i, filename in enumerate(filenames):
        file_path = os.path.join(directory, filename)
        img = img_as_float(imread(file_path))
        prepared = prepare_img(img, tg_size)
        result[i] = prepared
        shapes[i] = img.shape[:2]
    return result, shapes, filenames

def train_detector(train_gt, train_img_dir, fast_train=True):
    y = pd.DataFrame(train_gt).transpose().values
    data, shapes, filenames = get_data_shapes_filenames(train_img_dir)
    
    model = Sequential([
        Convolution2D(
            64, (3, 3),
            activation='relu',
            input_shape=(128, 128, 3),
            kernel_initializer='normal'),
        MaxPooling2D(
            pool_size=(2,2),
            strides=(2,2)),
        Convolution2D(
            128, (3, 3),
            activation='relu',
            kernel_initializer='normal'),
        MaxPooling2D(
            pool_size=(2,2),
            strides=(2,2)),
        Convolution2D(
            256, (3, 3),
            activation='relu',
            kernel_initializer='normal'),
        MaxPooling2D(
            pool_size=(2,2),
            strides=(2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.25),
        Dense(28)
    ])
        
    adam = Adam(lr=0.0003)
    model.compile(loss='mean_squared_error',
                optimizer=adam,
                metrics=['mean_absolute_error'])

    model.fit(data, y, epochs=1)
    return model
    
# returns dict: {filename -> [number]}
def detect(model, test_img_dir):
    data, shapes, filenames = get_data_shapes_filenames(test_img_dir)
    answers = []
    batch_size = 500
    for i in range((len(data) + batch_size - 1) // batch_size):
        answers.extend(model.predict(data[i*batch_size : min((i+1)*batch_size, len(data))]))
    return {filenames[i] : answers[i] * shapes[i, 0] for i in range(len(filenames))}
