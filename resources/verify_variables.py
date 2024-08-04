import os
from typing import Any, Union, Callable
import numpy as np
from resources.message import error_message, warning_message, method_menssage
import cv2
"""
El siguiente codigo contiene las funciones que verifican las variables que se ingresan de forma recurrente a los métodos
de las demás rutinas, por ejemplo, verifica los tipos de datos, rangos, existencia de rutas. El mismo, verifica posibles
errores y advertencias, apoyandose del modulo "message" y las funciones "error_message" y "warning_message" para detener
el código en caso de encontrar un error o solamante, enviar un mensaje de advertencia sin detener el codigo.
"""
class VerifyErrors():
    def __init__(self) -> None:
        """
        Contiene métodos que verifican diferentes condiciones obligatorios sobre las variables de entrada, como su tipo, 
        existencia, relación entre minimos y máximos, entre otras posibles. El flujo de ejecución se detiene de no validar 
        satisfactoriamente la verificación.
        """
        pass
    
    def check_type(self, var: Any, type: type, label: str) -> str:
        """
        Verifica que el tipo de dato ingresado coincide con el tipo de dato esperado.
        Args:
            var (Any):      La variable a verificar.
            type (type):    El tipo de dato con el que se verifica la variable.
            label (str):    Etiqueta de texto para identificar la variable en el texto de salida, en caso de haber error.
        
        Returns:
            str:    Mensaje de error o validación.
        """
        if not isinstance(var, type):
            return print(error_message(f'la variable "{label}" debe ser de tipo {type.__name__}.'))
        else:
            return print(f'  ● Verificación de tipo sobre la variable "{label}": ✅ .')

    def check_numeric_min_max(self, var_min:Union[int, float], var_max:Union[int, float], label_min:str, label_max:str) -> str:
        """
        Verifica que el dato ingresado se encuentra dentro de los rangos numéricos esperados.

        Args:
            var_min (Union[int, float]):      La variable más pequeña.
            var_max (Union[int, float]):      La variable más grande.
            label_min (str):    Etiqueta de texto para identificar la variable más pequeña en el texto de salida, en caso de haber error.
            label_max (str):    Etiqueta de texto para identificar la variable más grande en el texto de salida, en caso de haber error.
        
        Returns:
            str: Mensaje de error o validación.
        """
        if var_min >= var_max:
            return print(error_message(f'La variable "{label_min}" = {var_min} debe ser menor al número máximo.'))
        else:
            return print(f'  ● Verificación de rango sobre las variables "{label_min}" = {var_min} y "{label_max}" = {var_max}: ✅ .')

    def check_positive(self, var: Union[int, float, None], label:str) -> str:
        """
        Verifica que el dato ingresado no sea menor o igual a cero.

        Args:
            var (Union[int, float]): Variable numérica a verificar.
            label (str): Etiqueta de texto para identificar la variable en el texto de salida.
        
        Returns:
            str: Mensaje de error o validación. 
        """
        if var == None:
            return print(f'  ● Verificación de valor mayor a cero sobre la variable "{label}"={var}: ✅ .')
        elif var <= 0:
            return print(error_message(f'La variable "{label}" = {var} debe ser mayor a cero.'))
        else:
            return print(f'  ● Verificación de valor mayor a cero sobre la variable "{label}"={var}: ✅ .')
        
    def check_path(self, path:str) -> str:
        """
        Verifica que la ruta en cuestión exista.
        Args:
            path (str): Ruta a verificar.
        
        Returns:
            str: Mensaje de error o validación.
        """
        if not os.path.exists(path):
           return print(error_message(f'La ruta "{path}" no existe.'))
        else:
            return print(f'  ● Verificación de existencia de la ruta "{path}": ✅ .')
    
    def check_folder(self, path:str) -> str:
        """
        Verifica que la ruta en cuestión exista.
        Args:
            path (str): Ruta a verificar.
        
        Returns:
            str: Mensaje de error o validación.
        """
        if not os.path.isdir(path):
            return print(error_message(f'La ruta "{path}" debe ser una carpeta.'))
        else:
            return print(f'  ● Verificación de existencia de la carpeta "{path}": ✅ .')
    
    def check_file_tipe(self, path:str, files:Union[list[str], tuple[str]]) -> str:
        """
        Verifica que los archivos dentro de una lista son de tipo imágen de formato tif, jpg, jpeg, png, gif o bmp.

        Args:
            path (str):                             La ruta donde se encuentras las imágenes para construir del dataset.
            files (Union[list[str], tuple[str]]):   Iterable (lista o tupla) con los nombres de los archivos de imágenes junto con su extensión.
        
        Returns:
            str: Mensaje de error o validación.
        """
        extensions = ['.tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
        if not files:
            return error_message(f'No hay archivos dentro del iterable.')
        for file in files:
            pth_img = os.path.join(path+file)
            if not os.path.exists(pth_img):
                return print(error_message(f'El archivo "{pth_img}" no fue encontrado.'))
            ext     = pth_img[-4:].lower()
            if ext not in extensions:
                return print(error_message(f'Se detectó el archivo "{pth_img}" con un formato diferente a tif, jpg, jpeg, png, gif o bmp.'))
        return print(f'  ● Verificación de los tipos de archivos como imágenes: ✅ .')

    def check_dimension(self, dim:int, path:str, files:Union[list[str], tuple[str]]) -> str:
        """
        Verifica que la dimensión de las secciones del dataset es compatible con las imágenes que se usarán para obtener las secciones.
            ●   La dimensión no puede ser mayor a las resolución de la imágen original en alto o en ancho.
            ●   La división entre las resoluciones y la dimensión debe dar un residuo de 0.

        Args:
            dim (int):                              Dimensión de cada sección (mxm).
            path (str):                             Ruta donde se encuentras las imágenes para construir del dataset.
            files (Union[list[str], tuple[str]]):   Iterable (lista o tupla) con los nombres de los archivos de imágenes junto con su extensión.
        
        Returns:
            str: Mensaje de error o validación.
        """
        for file in files:
            pth_img = os.path.join(path+file)
            img     = cv2.imread(pth_img)
            w, h    = img.shape[1], img.shape[0]
            if (w%dim or h%dim) != 0:
                return print(error_message(f'La dimensión {dim} para las secciones no es compatible para imágenes de resolución ancho={w}px, alto={h}px.\nLa dimensión debe dar una división exacta (residuo 0), al dividirse por el alto y ancho de la resolución de las imágenes.'))
            elif (dim>=w or dim>=h):
                return print(error_message(f'La dimensión {dim} para las secciones no puede ser mayor a la resolución de ancho={w}px o alto={h}px de las imágnes.'))
            else:
                return print(f'  ● Verificación del valor de dimensión para las secciones del dataset: ✅ .')

    def check_arguments(self, arg:Any, types:list, var_label:str) -> str:
        """
        Verifica que el argumento sea uno de los valores añadido a la lista.

        Args:
            arg (Any):          Argumento a evaluar.
            types (list):       Lista con los valores posibles que puede adoptar el argumento.
            var_label (str):    Texto que identifica a la variable del argumento.
        
        Returns:
            str: Mensaje de error o validación.
        """
        if arg not in types:
            return print(error_message(f'La variable "{var_label}" debe tener alguno de los valores los valores {types}.'))
        else:
            return print(f'  ● Verificación de argumento para la variable {var_label}: ✅ .')

    def check_par(self, arg:Any, var_label:str) -> str:
        """
        Verifica que el argumento sea uno de los valores añadido a la lista.

        Args:
            arg (Any):       Argumento a evaluar.
            var_label (str): Texto que identifica a la variable del argumento.
        
        Returns:
            str: Mensaje de error o validación.
        """
        if not arg%2 == 0:
            return print(error_message(f'La variable "{var_label}" debe ser un numero par'))
        else:
            return print(f'  ● Verificación de número par para la variable {var_label}: ✅ .')

    def check_dim_layers(self, dim:str, number_layers:str) -> str:
        """
        Verifica que la dimensión de los datos de entrada sea factible para reducirse el número de veces que indica
        el número de capas en el encoder.

        Args:
            dim (str):           La dimensión m de los datos de entrada (m,m,3).
            number_layers (str): El número de capas con el que se hará el encoder.
        
        Returns:
            str: Mensaje de error o validación.
        """
        cheack_dim = dim
        for i in range(number_layers):
            cheack_dim /= 2
            if round(cheack_dim) < 3:         
                return print(error_message('El encoder tiene una cantidad capas superior a la esperada por la dimensión de los datos de entrenamiento.'))
        
        return print(f'  ● Verificación del valor de dimensión respecto a la cantidad de capas en el encoder: ✅ .')

    def check_dim_dataset(self, dataset:np.ndarray, dim:int) -> str:
        """
        Verifica que la dimensión de las imágenes del dataset en alto y ancho tenga el mismo valor que el ingresado en la variable dim.
        Además, verifica que el dataset tenga una dimensionalidad de arreglo esperada de 5.

        Args:
            dataset (np.ndarray): Arreglo de numpy con la información de los datos de entrenamiento.
            dim     (str): La dimensión m de los datos de entrada (m,m,3).
        Returns:
            str: Mensaje de error o validación.
        """
        dim1 = dataset.shape[1]
        dim2 = dataset.shape[2]
        if not (dim == dim1 and dim == dim2):
            return print(error_message(f'La dimensión de las imágenes del dataset = ({dim1},{dim2}) no es igual a la ingresada en la variable dim = {dim}'))
        elif not dataset.ndim == 4:
            return print(error_message(f'El dataset ingresado tiene una dimensión {dataset.ndim}, cuando se esperan 5 de la forma:\n(número de conjuntos, número de imágenes en un conjunto, ancho de la imagen, alto de la imágen, 3)'))
        else:
            return print(f'  ● Verificación de igualdad entre la dimensión de las imágenes del dataset y dimensión de la variable dim: ✅ .\n  ● Verificación de la dimensionalidad del dataset: ✅ .')

    def check_provided(self, lista:list, label:str, fun:Callable, label_fun:str) -> str:
        """
        Verifica que una lista de argumentos no tenga valores nulos (argumentos obligatorios para cada función).

        Args:
            list        (list): Lista con los valores a verificar.
            label       (str): Texto que describe lo que se desea hacer con la función.
            fun         (Callable): Función que se está evaluando.
            label_fun   (str): Texto que describe lo que hace la función.

        Returns:
            str: Mensaje de error o validación.
        """
        for arg in lista:
            if arg is None:
                return print(error_message(f'Un argumento tiene valor None, debe proporcionar un valor adecuado para este si desea {label}.'))
            else:
                pass
        return method_menssage(fun.__name__, label_fun)

class VerifyWarnings():
    def __init__(self) -> None:
        """
        Contiene métodos que verifican diferentes condiciones recomendadas sobre las variables de entrada. Se pausa 
        temporalmente el flujo de ejecución para notificar al usuario en caso de no validar satisfactoriamente la 
        verificación.
        """
        pass

    def check_limits(self, var:Union[int, float, None], l_min:Union[int, float], l_max:Union[int, float], label:str) -> str:
        """
        Verifica que una variable de tipo numérico se encuentre en determinado rango aconsejado de valores.

        Args:
            var (Union[int,float]):     Variable numérica a verificar.
            l_min (Union[int,float]):   Limite inferior con el que se evalua la variable.
            l_max (Union[int,float]):   Limite superior con el que se evalua la variable.
            label (str):                Etiqueta de texto para identificar la variable en el mensaje de salida.
        
        Returns:
            str: Mensaje de error o validación.
        """
        if var == None:
            return  print(f'  ● Verificación de límites para la variable "{label}": ✅ .')
        elif not l_min <= var <= l_max:
            return print(warning_message(f'Se recomienda que la variable "{label}" tenga valores entre {l_min} y {l_max}. No es obligatorio para la ejecución del algoritmo, pero puede afectar en los resultados del entrenamiento.'))
        else:
            return  print(f'  ● Verificación de límites para la variable "{label}": ✅ .')

    def check_resolutions(self, path:str, files:Union[list[str], tuple[str]]) -> str:
        """
        Verifica que las imágenes que se usarán para crear el dataset tienen la misma resolución.

        Args:
            path (str):                             La ruta donde se encuentras las imágenes para construir del dataset.
            files (Union[list[str], tuple[str]]):   Iterable (lista o tupla) con los nombres de los archivos de imágenes junto con su extensión.
        
        Returns:
            str: Mensaje de error o validación.
        """
        resolutions = []
        for file in files:
            pth_img = os.path.join(path+file)
            img     = cv2.imread(pth_img)
            w, h    = img.shape[1], img.shape[0]
            dim     = [w,h]
            if not dim in resolutions:
                resolutions.append(dim)
        num_res     = len(resolutions)
        if num_res > 1:
            return print(warning_message(f'Se encontraron imágenes con {num_res} diferentes resoluciones, esto puede provocar errores con la dimensión de los datos.'))
        else:
            return print(f'  ● Verificación de las resoluciones de las imágenes: ✅ .')