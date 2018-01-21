# Utilites for the model in this folder

from keras import backend as ks
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling3D, UpSampling2D
from keras.layers import Conv3D, Conv2D, AveragePooling2D, AveragePooling3D, Dense, Dropout, LeakyReLU
from keras.layers import MaxPooling2D, MaxPooling3D, Conv2DTranspose, Input
from keras.layers.core import Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import SGD, Adam
from plotter import LossAccPlotter
from keras import utils

import pywavefront as pw
import numpy as np
import os.path
import time
import glob
import math
import os

ks._BACKEND = 'theano'
ks.set_image_dim_ordering("th")

def update_progress(progress, step=0.001,string="",loss=False):
    #ignore this pls, pure bullshit, but it looks good in the terminal.
    print("\r"+string+" [{0}] {1}%".format('#'*int(progress/10 if loss else progress/500), int(progress + 1 if loss else progress/50)), end="")
    time.sleep(step)


def load_data():  
    if os.path.isfile('dataArray.npy'):
        print("Loading existing data...")
        data = np.load(open('dataArray.npy', 'rb'))
    else:
        print("Loading data...")  
        data=[]
        for i in range(0,5000):
            directory =  '/home/viktorv/Projects/MachineLearning/CiD/data/concept_processed/cube'+str(i)+'.obj00'
            data.append((pw.ObjParser(pw.Wavefront(directory), directory).vertices))
            update_progress(i)
        #endfor
        data = np.array(data)
        data.dump(open('dataArray.npy', 'wb'))
    #endif
    # print(data.shape)
    # data = data.reshape((data.shape[0], data.shape[1], data.shape[2], 1))
    # data = np.resize(data.shape, (5000, 9025, 3)).reshape((5000, 95, 95, 3))
    data = np.resize(data.shape, (5000, 9025, 3)).reshape((5000, 3, 95, 95))    
    
    print(data.shape)

    print("Loading complete!")

    return data


def generator_model(_1d=False):
    model = Sequential()
    #TODO: Try 1D Input and Output, idk.
    depth = 32
    dropout_rate = 0.4

    model.add(Dense(128, input_shape=(128,)))
    model.add(Reshape((128,1,1), input_shape=(128,)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))    

    model.add(Conv2DTranspose(depth*8,(4,4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))   

    model.add(Conv2DTranspose(depth*4,(4,4), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv2DTranspose(depth*2,(4,4), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Conv2DTranspose(depth,(5,5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Conv2DTranspose(int(depth/2), (5,5), strides=(2, 2)))    
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(3,(3,3)))
    model.add(Activation('tanh'))

    model.summary()

    return model


def discriminator_model(_1d=False):
    model = Sequential()

    depth = 32
    dropout_rate = 0.4

    model.add(Conv2D(depth, (3,3), input_shape=(3, 95, 95,)))
    model.add(BatchNormalization())  
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(depth*2, (3,3)))
    model.add(BatchNormalization())    
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(depth*4, (3,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(depth*8, (3,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))    
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.summary()
    return model


def generator_containing_discriminator(generator, discriminator):

    model = Sequential()
    model.add(generator)    
    discriminator.trainable = False
    model.add(discriminator)
    
    return model

def train(epochs, BATCH_SIZE, load=False):

    plotter = LossAccPlotter(title="DCGAN Adversary plot",
                         save_to_filepath="./plots/my_plot.png",
                         show_averages=True,
                         show_loss_plot=True,
                         x_label="Index")
    X_train = load_data()

    discriminator = discriminator_model()
    generator = generator_model()

    #tbCallBack = TensorBoard(log_dir='graph', histogram_freq=5, write_graph=True, write_images=True)
    #tbCallBack.set_model(generator)

    if load:
        generator.load_weights('goodgenerator.h5')
        discriminator.load_weights('gooddiscriminator.h5')
    
    discriminator_on_generator = generator_containing_discriminator(generator, discriminator)

    d_optim = SGD(lr=0.0002, momentum=0.7)
    g_optim = Adam(lr=0.0002, beta_1=0.5)

    generator.compile(loss='binary_crossentropy', optimizer="adam")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    noise = np.zeros((BATCH_SIZE, 128))

    for epoch in range(epochs):

        for index in range(int(X_train.shape[0]/BATCH_SIZE)):

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.normal(-1, 1, 128)

            batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]

            generated = generator.predict(noise, verbose=0)

            X = np.concatenate((batch, generated))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.normal(-1, 1, 128)

            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)
            discriminator.trainable = True

            update_progress(index, loss=True, string="Epoch: %d Batch: %d Dloss: %f Gloss: %f" % (epoch, index, d_loss, g_loss))
            plotter.add_values(index + BATCH_SIZE*epoch, loss_train=d_loss, loss_val=g_loss)

            if epoch % 10 == 9:
                generator.save_weights('goodgenerator.h5', True)
                discriminator.save_weights('gooddiscriminator.h5', True)
        print()


def obj_wrapper(coords, name="object"):
    lines = ""
    for i in range(0, len(coords)):
        lines += "v " + str(coords[i,0]) + " " + str(coords[i,1]) + " " + str(coords[i,2]) + " #" + str(i + 1) + "\n"
    
    return lines


def generate(BATCH_SIZE,name="generated"):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="adam")
    generator.load_weights('goodgenerator.h5')

    noise = np.zeros((BATCH_SIZE, 128))
    a = np.random.normal(-1, 1, 128)
    b = np.random.normal(-1, 1, 128)
    grad = (b - a) / BATCH_SIZE

    for i in range(BATCH_SIZE):
        noise[i, :] = np.random.normal(-1, 1, 128)

    generated = generator.predict(noise)

    for i, pointcloud in enumerate(generated):
        pointcloud = pointcloud.reshape(9025, 3)
        file = open('./generated_data/%s%s.obj'%(name,i), "w")
        file.write(obj_wrapper(pointcloud))
        file.close()
        
        
