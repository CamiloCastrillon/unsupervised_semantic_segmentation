import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras._tf_keras.keras.models import load_model
import numpy as np
import cv2
from resources.create_dataset import GenDataAutoencoder as gda
import matplotlib.pyplot as plt
from resources.general import create_path_save


gda = gda(None, None, None)
path_img            = 'C:/camilo/uss/data/img.tif'
path_model          = 'C:/camilo/uss/models/full_auto_encoder/full_auto_encoder_pruebas.keras'
path_save_predicts  = 'C:/camilo/uss/predicts/full_auto_encoder/'

dim = 50
full_auto   = load_model(path_model)
img         =  cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
w, h  = img.shape[1], img.shape[0]

secciones = gda.make_data(img, w, h, dim)
secciones_array = np.array(secciones)

predicciones = full_auto.predict(secciones_array)

np.save(path_save_predicts, predicciones)

large_image = np.zeros((h, w, 3), dtype=secciones_array.dtype) #---------- Inicializa la imagen vacía (arreglo de las dimensiones de la imagen original lleno de ceros)
print(large_image[0])
index = 0 #------------------------------------------------------ Define el indice para traer cada imagen de las predicciones del full autoencoder a la imagen completa
for row in range (0, h, dim): #---------------------------------- Itera sobre las filas que componen la imagen completa a traves de las imágenes pequeñas
  for col in range(0, w, dim):  #-------------------------------- Itera sobre las columnas que componen la imagen completa a traves de las imágenes pequeñas
    large_image[row:row+dim, col:col+dim] = predicciones[index] # Reescribe la imagen vacía con la imagen que corresponde proveniente de las predicciones
    index += 1  #------------------------------------------------ Aumenta el contador del índice para traer la imagen siguiente en la proxima iteración
print(large_image[0])

plt.rcParams['font.family'] = 'Times New Roman'
fig, axis = plt.subplots(1, 2, figsize=(8, 6)) # Define la figura de 1 fila, dos columnas y tamaño 5x8

# Muestra la imagen original
axis[0].imshow(img)
axis[0].set_title('Imagen Original', fontdict={'weight': 'bold', 'size': 12})
axis[0].axis('off')

# Muestra la imagen reconstruida
axis[1].imshow(large_image)
axis[1].set_title('Imagen Reconstruida', fontdict={'weight': 'bold', 'size': 12})
axis[1].axis('off')

plt.show()