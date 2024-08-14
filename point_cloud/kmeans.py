import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import pickle

folder_imgs         = 'C:/camilo/imgs/aguacate/'
files_imgs          =  os.listdir(folder_imgs)
num_clusters        = 3
images_flatten    = []

#Creando dataset
print('Creando dataset.\n')
for name_img in files_imgs:
    entire_path = folder_imgs+name_img
    img         = cv2.imread(entire_path)
    w, h, _     = img.shape
    img_flatten = img.reshape(-1, 3)
    images_flatten.append(img_flatten)
images_flatten_array = np.vstack(images_flatten)

# Entrenando modelo
print('Entrenando modelo.\n')
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(images_flatten_array)

# Guardar el modelo entrenado
with open('C:/camilo/imgs/models/kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)
print('Modelo KMeans guardado como kmeans_model.pkl')
