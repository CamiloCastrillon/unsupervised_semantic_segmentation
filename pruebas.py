import tensorflow as tf
from keras._tf_keras.keras import models

model =models.Sequential(name='full_autoencoder')

print(type(model))

if isinstance(model, models.Sequential):
    print('correcto')
else:
    print('incorrecto')