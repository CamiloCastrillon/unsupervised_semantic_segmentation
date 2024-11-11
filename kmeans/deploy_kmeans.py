import cv2
import os
import numpy as np
import pickle

folder_save = 'C:/camilo/trabajo_de_grado/imgs/aguacate_kmeans/'
path_model  = 'C:/camilo/trabajo_de_grado/imgs/models/kmeans_model.pkl'
folder_imgs = 'C:/camilo/trabajo_de_grado/imgs/dataset/'

files_imgs  =  os.listdir(folder_imgs)

# Cargar el modelo entrenado
print('Cargando modelo.')
with open(path_model, 'rb') as file:
    kmeans = pickle.load(file)

#Implementando modelo
print('Implementando modelo.\n')
for name_img in files_imgs:
    entire_path = folder_imgs+name_img
    img         = cv2.imread(entire_path)
    print(img.shape)
    h, w, _     = img.shape
    img_flatten = img.reshape(-1, 3)

    # Obtener las etiquetas de los clusters y los centros de los clusters
    labels = kmeans.predict(img_flatten)

    # Reshape de las etiquetas para que coincidan con la forma original de la imagen (w, h)
    labels_reshaped = labels.reshape(h, w)

    # Crear una nueva matriz con la imagen RGB y las etiquetas como la cuarta banda
    img_with_labels = np.dstack((img, labels_reshaped))

    # Guardar la imagen con la cuarta banda en formato npy
    save_path = os.path.join(folder_save, f'{name_img[:-4]}_data.npy')
    np.save(save_path, img_with_labels)
    
    """
    # Crear un array para guardar los datos RGB y la clase
    data = np.empty((h, w, 4), dtype=np.uint8)  # 4 canales: R, G, B, class
    
    # Rellenar el array con los valores RGB y la clase
    for i in range(len(labels)):
        y = i // w
        x = i % w
        data[y, x, 0] = img_flatten[i, 0]  # R
        data[y, x, 1] = img_flatten[i, 1]  # G
        data[y, x, 2] = img_flatten[i, 2]  # B
        data[y, x, 3] = labels[i]           # Clase

    # Guardar la imagen como archivo NumPy
    path_npy = folder_save + name_img[:-4] + '_data.npy'
    np.save(path_npy, data)
    """
    print(f'    ->Datos de {name_img[:-4]} guardados en {save_path}.')