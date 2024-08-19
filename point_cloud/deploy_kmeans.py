import cv2
import os
import numpy as np
import pickle
from PIL import Image


folder_save = 'C:/camilo/imgs/aguacate_kmeans/'
path_model  = 'C:/camilo/imgs/models/kmeans_model.pkl'
folder_imgs         = 'C:/camilo/imgs/aguacate/'

files_imgs          =  os.listdir(folder_imgs)

# Cargar el modelo entrenado
print('Cargando modelo.')
with open(path_model, 'rb') as file:
    kmeans = pickle.load(file)

#Implementando modelo
print('Implementando modelo.\n')
for name_img in files_imgs:
    entire_path = folder_imgs+name_img
    img         = cv2.imread(entire_path)
    w, h, _     = img.shape
    img_flatten = img.reshape(-1, 3)

    # Obtener las etiquetas de los clusters y los centros de los clusters
    labels = kmeans.predict(img_flatten)
    # Crea la imagen rgba
    img_class = labels.reshape(w, h).astype(np.uint8)
    #img_class = np.dstack((img, labels.reshape(w, h).astype(np.uint8)))
    path_save = folder_save + name_img[:-4] + '_kmeans.tiff'
    cv2.imwrite(path_save, img_class)
    print(f'    ->Imagen {name_img[:-4]}_kmeans.tiff guardada.')