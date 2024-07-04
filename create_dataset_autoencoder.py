"""
El siguiente código tiene como objetivo generar un conjunto de datos a partir de imágenes
localizadas en un carpeta, que sean óptimos para el entrenamiento del full autoencoder.

Autor: Juan Camilo Navia Castrillón
Fecha: Por definir
"""

import os
import cv2
import numpy as np

class GenDataAutoencoder:
    """
    Genera el conjunto de datos
    """
    def __init__(self):
        self.pth        = os.path.dirname(os.path.abspath(__file__))        # Obtiene la ruta de la carpeta actual
        self.pth_data   = os.path.join(self.pth, 'data')                    # Obtiene la ruta de la carpeta donde se encuentran las imagenes de entrenamiento
        self.imgs       = tuple(os.listdir(self.pth_data))                  # Crea una tupla con las rutas de las imágenes
        self.pth_save   = os.path.join(self.pth, 'datasets', 'autoencoder') # Crea la ruta donde se guardará el dataset para entrenar el autoencoder

    def get_imgs(self):
        """
        Obtiene la ruta y carga las imágenes una por una, abriendola y obteniendo sus dimenciones.
        """
        for img in self.imgs:                                               # Itera subre la tupla con los nombres de las imágenes
            pth_img = os.path.join(self.pth_data, img)                      # Obtiene la ruta de la imagen            
            img     = cv2.cvtColor(cv2.imread(pth_img), cv2.COLOR_BGR2RGB)  # Abre la imágen y transforma de BGR a RGB
            w, h    = img.shape[1], img.shape[0]                            # Obtiene el ancho (w) y el alto (h) en pixeles
            yield img, w, h
    
    def make_data(img, h, w, dim):
        """
        Divide ima imagen de entrada en secciones de imágenes mas pequeñas, dados los parámetros.
         
            img: Imágen (nd.array).
            h: Alto de la imágen (int).
            w: Ancho de la imágen (int).
            dim: Dimensión de la sección (int).
        """
        sections = []
        
        for row in range(0, h, dim):
            for col in range(0, w, dim):
                section = img[row:row+dim, col:col+dim]     # Itera sobre la imagen de entrada, extrayendo secciones de dimensión (dim,dim)
                section = section.reshape(dim, dim, 3)      # Redimensiona la sección a la especificada y con 3 canales de profundidad (RGB)
                section = section.astype('float32')/255.0   # Asegura el tipo de dato float32 y normaliza los datos de los pixeles
                sections.append(section)                    # Cada sección individual (section) se añade a una lista de secciones (sections)

        return sections
        
    def define_data(name, stack):
        """
        Se encarga de transformar una lista en un arreglo de guardarlo en un archivo npy, 
        en la ruta especificada por el parámetro name, se espera que se ingrese una lista con las
        secciones de todas las imágenes contenidas en la carpeta de dataset.
        """
        dataset = np.array(stack)       # Transforma el objeto list a nd.array
        np.save(name+'.npy', dataset)   # Guarde el arreglo