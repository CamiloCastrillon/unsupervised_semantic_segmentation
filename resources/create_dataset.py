"""
El siguiente código tiene como objetivo generar un conjunto de datos a partir de imágenes
localizadas en un carpeta, que sean óptimos para el entrenamiento del full autoencoder.

Autor: Juan Camilo Navia Castrillón
Fecha: Por definir
"""
import os
import cv2
import numpy as np
from datetime import datetime
from resources.message import method_menssage
from resources.verify_variables import VerifyErrors as ve, VerifyWarnings as vw
from resources.general import create_path_save

class GenDataAutoencoder:
    """
    Genera el conjunto de datos en formato npy, compuesto de trozos de imágen con dimensión mxm.

    Args:
        dim (int):      Dimensión m de los datos del conjunto de datos.
        pth_data (str): Ruta de la carpeta con las imágenes para crear el dataset.
        pth_save (str): Ruta de la carpeta con las imágenes para guardar el dataset.
    
    Returns:
        None: No se espera argumento de salida.
    """
    def __init__(self, dim:int, pth_data:str, pth_save:str) -> None:
        self.dim        = dim                               # Dimensión de las secciones de los datos de entrenamiento
        self.pth_data   = pth_data                          # Obtiene la ruta de la carpeta donde se encuentran las imagenes de entrenamiento
        self.imgs       = tuple(os.listdir(self.pth_data))  # Crea una tupla con las rutas de las imágenes
        self.pth_save   = pth_save                          # Crea la ruta donde se guardará el dataset para entrenar el autoencoder
 
    def cheack_values(self):
        """
        Verifica los posibles errores y advertencias al ingresar los argumentos de la clase GenDataAutoencoder.

        Esta función no espera argumentos ni devuelve valores.
        """
        method_menssage(self.cheack_values.__name__, 'Verifica los posibles errores y advertencias al ingresar los argumentos de la clase GenDataAutoencoder')
        # Evalua la las rutas de los datos y guardado
        ve().check_path(self.pth_data)
        ve().check_path(self.pth_save)
        ve().check_folder(self.pth_data)
        ve().check_folder(self.pth_save)

        # Evalua que existan solo archivos de imágenes en la ruta de datos
        ve().check_file_tipe(self.pth_data, self.imgs)

        # Evalua la variable dim
        label_dim = 'Dimensión de las imágenes'
        ve().check_type(self.dim, int, label_dim)
        ve().check_positive(self.dim, label_dim)
        vw().check_limits(self.dim, 25, 100, label_dim)

        # Evalua si hay imágenes con diferente dimensionalidad
        vw().check_resolutions(self.pth_data, self.imgs)
        # Evalua que la dimensión sea posible de aplicar en las imágenes (que no sea mayor y además que sea divisible)
        ve().check_dimension(self.dim, self.pth_data, self.imgs)

    def get_imgs(self):
        """
        Obtiene la ruta y carga las imágenes una por una, abriendola y obteniendo sus dimenciones.

        Returns:
            img (np.ndarray):   Imágen a seccionar.
            h (int):            Alto de la imágen en píxeles.
            w (int):            Ancho de la imágen en píxeles.
        """
        method_menssage(self.get_imgs.__name__, 'Obtiene imágen por imágen de la ruta con los datos, así como la dimensión ancho y alto en pixeles de las mismas')
        for img in self.imgs:                                               # Itera subre la tupla con los nombres de las imágenes
            pth_img = os.path.join(self.pth_data, img)                      # Obtiene la ruta de la imagen            
            img     = cv2.cvtColor(cv2.imread(pth_img), cv2.COLOR_BGR2RGB)  # Abre la imágen y transforma de BGR a RGB
            w, h    = img.shape[1], img.shape[0]                            # Obtiene el ancho (w) y el alto (h) en pixeles
            yield img, w, h
    
    def make_data(self, img:np.ndarray, w:int, h:int, dim:int) -> list:
        """
        Divide ima imagen de entrada en secciones de imágenes mas pequeñas, dados los parámetros.
        
        Args:
            img (np.ndarray):   Imágen a seccionar.
            h (int):            Alto de la imágen en píxeles.
            w (int):            Ancho de la imágen en píxeles.
            dim (int):          Dimensión de cada sección (mxm).
        
        Returns:
            sections (list):    Lista con cada una de las secciones de la imágen.
        """
        method_menssage(self.make_data.__name__, 'Crea el conjunto de datos dividiendo cada imágen en trozos de imágen más pequeños con la dimensión dada')
        sections = []
        for row in range(0, h, dim):
            for col in range(0, w, dim):
                section = img[row:row+dim, col:col+dim]     # Itera sobre la imagen de entrada, extrayendo secciones de dimensión (dim,dim)
                section = section.reshape(dim, dim, 3)      # Redimensiona la sección a la especificada y con 3 canales de profundidad (RGB)
                section = section.astype('float32')/255.0   # Asegura el tipo de dato float32 y normaliza los datos de los pixeles
                sections.append(section)                    # Cada sección individual (section) se añade a una lista de secciones (sections)
        return sections
        
    def save_data(self, pth_save:str, stack:list) -> None:
        """
        Se encarga de transformar la lista con la secciones creadas en un arreglo de guardarlo en un archivo npy, 
        en la ruta especificada.

        Args:
            pth_save (str): Ruta de la carpeta con las imágenes para guardar el dataset.
            stack (list):   Es la lista que contiene cada una de las listas "sections" con las secciones de cada imágen.
        
        Returns:
            None: No se espera argumento de salida.
        """
        method_menssage(self.save_data.__name__, 'Convierte el conjunto de datos a un arreglo de numpy y lo guarda en la ruta dada para esto')
        dataset = np.array(stack)   # Transforma el objeto list a nd.array
        np.save(pth_save, dataset)  # Guarde el arreglo

    def gen_data(self):
        """
        Crea un flujo de trabajo para crear el conjunto de datos aplicando los metodos de esta clase.

        Esta función no devuelve ningún argumento ni devuelve ningún valor.
        """
        method_menssage(self.gen_data.__name__, 'Ejecuta el flujo de trabajo que genera el conjunto de datos y lo guarda')
        self.cheack_values()
        stack = []
        for data in self.get_imgs():
            img, w, h   = data                                                      # Obtiene los datos para invocar make_data
            sections    = self.make_data(img, w, h, self.dim)                       # Genera las secciones (muestras)
            stack.append(sections)                                                  # Se ingresa cada sección a una lista
        pth_save = create_path_save(self.pth_save, f'dataset_dim{self.dim}', 'npy') # Define la ruta donde se guardará el archivo
        self.save_data(pth_save, stack)                                             # Todas las secciones se guardan como ndarray
        print(f'\nDataset generado con éxito en "{pth_save}".\n')