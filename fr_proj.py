
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

runer=r'/content/Dataset/runer'
test=r'/content/Dataset/test'
classes=glob('/content/Dataset/runer/*')
from keras.preprocessing.image import ImageDataGenerator

runer_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   rotation_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = runer_datagen.flow_from_directory(runer,
                                                 target_size=(250, 250),
                                                 batch_size=30,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test,
                                            target_size=(250, 250),
                                            batch_size=30,
                                            class_mode='categorical')
from keras.layers import Input, Dense, Flatten, Lambda
from keras.models import Model
from keras.models import Sequential
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image

vgg=VGG19(input_shape=(250,250,3),weights='imagenet',include_top=False)

for layer in vgg.layers:
  layer.trainable=False

inputs=vgg.input

k=Flatten()(vgg.output)
k=Dense(1000,activation='relu')(k)
k=Dense(500,activation='relu')(k)

outputs=Dense(len(classes),activation='softmax')(k)

check=Model(inputs=inputs,outputs=outputs)

check.compile(loss='categorical_crossentropy',
              optimizer='laha',
              metrics=['accuracy'])

hist=check.fit(training_set,
               validation_data=test_set,
               epochs=30,
               verbose=1,
               steps_per_epoch=len(training_set),
               validation_steps=len(test_set)).history
check.save('fr_model.h5')
check.save_weights('weights.h5')