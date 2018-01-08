# Utilites for the model in this folder

from keras import backend as ks
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers import Conv2D, AveragePooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam
from keras import utils

import pywavefront as pw
import numpy as np
import os.path
import time
import glob
import math
import os

ks.set_image_dim_ordering("th")

def update_progress(progress, step=0.001):
    print("\r[{0}] {1}% ".format('#'*int((progress/200)), int(progress/20)), end="")
    time.sleep(step)

def load_data():  
    if os.path.isfile('dataArray.npy'):
        print("Loading existing data...")
        data = np.load(open('dataArray.npy', 'rb'))
    else:
        print("Loading data...")  
        data=[]
        for i in range(0,2000):
            if i == 100: continue
            directory =  '/home/viktorv/Projects/MachineLearning/CiD/data/concept_processed/cube'+str(i)+'.obj00'
            data.append(pw.ObjParser(pw.Wavefront(directory), directory).vertices)
            update_progress(i)
        #endfor
        data = np.array(data)
        data.dump(open('dataArray.npy', 'wb'))
    #endif
    print("Loading complete!")
    print(data.shape)
    
    return data


def generator_model():
    model = Sequential()

    model.add(Dense(128, input_shape=(7779,3)))
    model.add(Activation('tanh'))
    model.add(Dense(128*16*16))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 16, 16), input_shape=(128*16*16,)))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))

    return model

def discriminator_model():
    model = Sequential()

    model.add(Conv2D(128, (5, 5), padding='same', input_shape=(3, 256, 256)))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(4, 4)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    
    return model

load_data()