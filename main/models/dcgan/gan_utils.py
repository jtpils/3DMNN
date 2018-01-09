# Utilites for the model in this folder

from keras import backend as ks
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D, UpSampling2D
from keras.layers import Conv3D, Conv2D, AveragePooling2D, AveragePooling3D, Dense, Dropout, LeakyReLU
from keras.layers import MaxPooling2D, MaxPooling3D
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

ks.set_image_dim_ordering("tf")

def update_progress(progress, step=0.001,string="",loss=False):
    print("\r"+string+" [{0}] {1}%                 ".format('#'*int(progress/10 if loss else progress/200), int(progress + 1 if loss else progress/20)), end="")
    time.sleep(step)


def load_data():  
    if os.path.isfile('dataArray.npy'):
        print("Loading existing data...")
        data = np.load(open('dataArray.npy', 'rb'))
    else:
        print("Loading data...")  
        data=[]
        for i in range(0,2000):
            directory =  '/home/viktorv/Projects/MachineLearning/CiD/data/concept_processed/cube'+str(i)+'.obj00'
            data.append((pw.ObjParser(pw.Wavefront(directory), directory).vertices))
            update_progress(i)
        #endfor
        data = np.array(data)
        data = np.resize(data.shape, (2000, 9216, 3)).reshape((2000, 96, 96, 3))
        data.dump(open('dataArray.npy', 'wb'))
    #endif
    print("Loading complete!")

    return data

def generator_model():
    model = Sequential()

    depth = 128
    model.add(Dense(depth, input_shape=(512,)))
    model.add(LeakyReLU())

    model.add(Dense(128*8*8))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((8, 8, 128), input_shape=(128*8*8,)))
    model.add(UpSampling2D(size=(3, 3)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(LeakyReLU())
    model.add(UpSampling2D(size=(4, 4)))
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(Activation("tanh"))

    return model

def discriminator_model():
    model = Sequential()

    depth = 128
    model.add(Conv2D(depth, (5, 5), padding='same', input_shape=(96, 96, 3, )))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(depth*2, (5, 5)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(depth*4, (5, 5)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(depth*4))
    model.add(LeakyReLU())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    discriminator.trainable = False
    model.add(generator)
    model.add(discriminator)
    
    return model

def train(epochs, BATCH_SIZE, load=False):

    X_train = load_data()

    discriminator = discriminator_model()
    generator = generator_model()

    if load:
        generator.load_weights('goodgenerator.h5')
        discriminator.load_weights('gooddiscriminator.h5')
    
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)

    generator.compile(loss='binary_crossentropy', optimizer="SGD")

    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)

    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    noise = np.zeros((BATCH_SIZE, 512))

    for epoch in range(epochs):
        #print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))

        for index in range(int(X_train.shape[0]/BATCH_SIZE)):

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 512)

            batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            generated = generator.predict(noise, verbose=0)

            X = np.concatenate((batch, generated))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 512)

            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)
            discriminator.trainable = True

            update_progress(index, loss=True, string="Epoch: %d Batch: %d Dloss: %f Gloss: %f" % (epoch, index, d_loss, g_loss))        

            if epoch % 10 == 9:
                generator.save_weights('goodgenerator.h5', True)
                discriminator.save_weights('gooddiscriminator.h5', True)
        print()

def cube_obj_wrapper(coords, name="object"):
    lines = ""
    for i in range(0, len(coords)):
        lines += "v " + str(coords[i,0]) + " " + str(coords[i,1]) + " " + str(coords[i,2]) + " #" + str(i + 1) + "\n"
    
    return lines


def generate(BATCH_SIZE):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('goodgenerator.h5')
    noise = np.zeros((BATCH_SIZE, 512))
    a = np.random.uniform(-1, 1, 512)
    b = np.random.uniform(-1, 1, 512)
    grad = (b - a) / BATCH_SIZE
    for i in range(BATCH_SIZE):
        noise[i, :] = np.random.uniform(-1, 1, 512)

    generated_images = generator.predict(noise)
    #image = combine_images(generated_images)

    for pointcloud in generated_images:
        pointcloud=pointcloud.reshape(9216,3)
        file = open("./generated.obj", "w")
        file.write(cube_obj_wrapper(pointcloud))
        file.close()
        
        
