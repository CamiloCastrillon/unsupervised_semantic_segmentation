"""
El siguiente código tiene como objetivo generar un conjunto de datos a partir de imágenes
localizadas en un carpeta, que sean óptimos para el entrenamiento del full autoencoder.

Autor: Juan Camilo Navia Castrillón
Fecha: Por definir
"""

import os
import cv2
import numpy as np
from resources.message import error_message, warning_message, method_menssage

class GenDataAutoencoder:
    """
    Genera el conjunto de datos
    """
    def __init__(self, dim, pth_data, pth_save):
        self.dim        = dim                               # Dimensión de las secciones de los datos de entrenamiento
        self.pth_data   = pth_data                          # Obtiene la ruta de la carpeta donde se encuentran las imagenes de entrenamiento
        self.imgs       = tuple(os.listdir(self.pth_data))  # Crea una tupla con las rutas de las imágenes
        self.pth_save   = pth_save                          # Crea la ruta donde se guardará el dataset para entrenar el autoencoder
 
    def cheack_values(self):
        """
        Evalua los posibles errores al ingresar los argumentos de la clase
        """
        # Evalua la variable dim
        if not isinstance(self.dim, int):
            error_message('La dimensión de los datos debe ser de tipo entero.')
        elif 0 < self.dim < 25:
            warning_message('La dimensión de los datos es demasiado baja, se sujiere un valor mínimo de 25. Tenga en cuenta la dimensión de las imágenes.')
        elif self.dim > 200:
            error_message('La dimensión de los datos es demasiado alta, se sujiere un valor menor a 200. Tenga en cuenta la dimensión de las imágenes.')
        elif self.dim < 0:
            error_message('La dimensión de los datos no puede ser negativa.')

        # Evalua la las rutas de los datos y guardado
        if not os.path.exists(self.pth_data):
            error_message(f'La ruta {self.pth_data} no existe.')
        elif not os.path.exists(self.pth_save):
            error_message(f'La ruta {self.pth_save} no existe.')
        elif not os.path.isdir(self.pth_data):
            error_message(f'La ruta {self.pth_data} debe ser una carpeta.')
        elif not os.path.isdir(self.pth_save):
            error_message(f'La ruta {self.pth_save} debe ser una carpeta.')
        
        # Evalua que existan solo archivos de imágenes en la ruta de datos
        extensions = ['.tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
        for img in self.imgs:                                               # Itera subre la tupla con los nombres de las imágenes
            pth_img = os.path.join(self.pth_data, img)
            ext = os.path.splitext(pth_img)[1].lower()
            if ext not in extensions:
                error_message(f'La ruta {pth_img} debe ser una imagen del formato tif, jpg, jpeg, png, gif o bmp.')

    def get_imgs(self):
        """
        Obtiene la ruta y carga las imágenes una por una, abriendola y obteniendo sus dimenciones.
        """
        method_menssage(self.get_imgs.__name__)
        for img in self.imgs:                                               # Itera subre la tupla con los nombres de las imágenes
            pth_img = os.path.join(self.pth_data, img)                      # Obtiene la ruta de la imagen            
            img     = cv2.cvtColor(cv2.imread(pth_img), cv2.COLOR_BGR2RGB)  # Abre la imágen y transforma de BGR a RGB
            w, h    = img.shape[1], img.shape[0]                            # Obtiene el ancho (w) y el alto (h) en pixeles
            yield img, w, h
    
    def make_data(self, img, w, h, dim):
        """
        Divide ima imagen de entrada en secciones de imágenes mas pequeñas, dados los parámetros.
         
            img: Imágen (nd.array).
            h: Alto de la imágen (int).
            w: Ancho de la imágen (int).
            dim: Dimensión de la sección (int).
        """
        method_menssage(self.make_data.__name__)
        sections = []
        for row in range(0, h, dim):
            for col in range(0, w, dim):
                section = img[row:row+dim, col:col+dim]     # Itera sobre la imagen de entrada, extrayendo secciones de dimensión (dim,dim)
                section = section.reshape(dim, dim, 3)      # Redimensiona la sección a la especificada y con 3 canales de profundidad (RGB)
                section = section.astype('float32')/255.0   # Asegura el tipo de dato float32 y normaliza los datos de los pixeles
                sections.append(section)                    # Cada sección individual (section) se añade a una lista de secciones (sections)

        return sections
        
    def save_data(self, pth_save, stack):
        """
        Se encarga de transformar una lista en un arreglo de guardarlo en un archivo npy, 
        en la ruta especificada por el parámetro name, se espera que se ingrese una lista con las
        secciones de todas las imágenes contenidas en la carpeta de dataset.
        """
        self.method_menssage(self.save_data.__name__)
        dataset = np.array(stack)   # Transforma el objeto list a nd.array
        np.save(pth_save, dataset)  # Guarde el arreglo

    def gen_data(self):
        """
        Crea un flujo de trabajo para crear el conjunto de datos aplicando los metodos de esta clase
        """
        self.cheack_values()
        stack = []
        for data in self.get_imgs():
            img, w, h = data                                                # Obtiene los datos para invocar make_data
            sections = self.make_data(img, w, h, self.dim)                  # Genera las secciones (muestras)
            stack.append(sections)                                          # Se ingresa cada sección a una lista
        pth_save = os.path.join(self.pth_save, f'dataset{self.dim}.npy')    # Crea la ruta de guardado con el archivo npy
        self.save_data(pth_save, stack)                                   # Todas las secciones se guardan como ndarray
        print(f'\nDataset generado con éxito en "{pth_save}".\n')