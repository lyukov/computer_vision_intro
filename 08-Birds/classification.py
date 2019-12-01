import os
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras.applications import VGG16
import pandas as pd
import numpy as np

image_size = 224

def train_classifier(train_gt, train_img_dir, fast_train=True):
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False

    model = models.Sequential()

    model.add(vgg_conv)

    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(50, activation='softmax'))

    model.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(lr=0.0003)
    )
    
    traindf = pd.DataFrame.from_dict(dict(
        filename=list(train_gt.keys()),
        class_id=list(map(lambda x: train_gt[x], train_gt))
    )).astype(str)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect'
    )

    generator_params = dict(
            directory=data_dir,
            x_col="filename",
            y_col="class_id",
            target_size=(image_size, image_size),
            batch_size=2,
            class_mode='categorical'
    )

    train_generator = datagen.flow_from_dataframe(
            dataframe=traindf,
            **generator_params
    )
    
    model.fit_generator(
        train_generator,
        steps_per_epoch=2,
        epochs=1
    )
    
    return model

def classify(model, test_img_dir):
    filenames = sorted(os.listdir(test_img_dir))
    
    test_generator = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rescale=1./255
    ).flow_from_dataframe(
        dataframe=pd.DataFrame(
            {'filename': filenames}
        ),
        directory=test_img_dir,
        shuffle=False,
        class_mode='input',
        x_col="filename",
        target_size=(image_size, image_size),
        batch_size=25
    )
        
    prediction = model.predict_generator(test_generator)
    answers = prediction.argmax(axis=1)
    
    return {
        fname: answer
        for fname, answer in zip(filenames, answers)
    }
