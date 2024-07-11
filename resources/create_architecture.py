import tensorflow as tf
from keras._tf_keras.keras import layers, optimizers, initializers, models, regularizers
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping

def random_choice(self):
    pass

def create_arch_full_auto(self, kernels, dim, number_layers):
    full_auto = models.Sequential(name='full_autoencoder')
    for lay in range(number_layers):
        if lay == 0 :
            full_auto.add(layers.Conv2D(kernels, (3, 3), activation='relu', padding='same', input_shape=(dim, dim, 3)))
            full_auto.add(layers.MaxPooling2D((2, 2)))
            kernels *= 2
        elif lay == number_layers-1:
            full_auto.add(layers.Conv2D(kernels, (3, 3), activation='relu', padding='same'))
            kernels -= kernels/2
        else:
            full_auto.add(layers.Conv2D(kernels, (3, 3), activation='relu', padding='same'))
            full_auto.add(layers.MaxPooling2D((2, 2)))
            kernels *= 2
    for lay in range(number_layers):
        if lay == number_layers-1:
            full_auto.add(layers.Conv2DTranspose(kernels, (2, 2), activation='relu'))
            full_auto.add(layers.UpSampling2D((2,2)))
            full_auto.add(layers.Conv2DTranspose(3, (3, 3), activation='relu', padding='same'))
        else:
            full_auto.add(layers.Conv2DTranspose(kernels, (3, 3), activation='relu', padding='same'))
            full_auto.add(layers.UpSampling2D((2,2)))
        kernels -= kernels/2
