import os
from typing import Any, Union
from resources.message import error_message, warning_message
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
            return error_message(f'la variable "{label}" = {var} debe ser de tipo {type.__name__}.')
        else:
            return (f'  ● Verificación de tipo sobre la variable "{label}" = {var}: ✔.')

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
            error_message(f'La variable "{label_min}" = {var_min} debe ser menor al número máximo.')
        else:
            return (f'  ● Verificación de rango sobre las variables "{label_min}" = {var_min} y "{label_max}" = {var_max}: ✔.')

    def check_positive(self, var: Union[int, float], label:str) -> str:
        """
        Verifica que el dato ingresado no sea menor o igual a cero.

        Args:
            var (Union[int, float]): Variable numérica a verificar.
            label (str): Etiqueta de texto para identificar la variable en el texto de salida.
        
        Returns:
            str: Mensaje de error o validación. 
        """
        if var <= 0:
            return error_message(f'La variable "{label}" = {var} debe ser mayor a cero.')
        else:
            return (f'  ● Verificación de valor mayor a cero sobre la variable "{label}"={var}: ✔.')
        
    def check_path(self, path:str) -> str:
        """
        Verifica que la ruta en cuestión exista.
        Args:
            path (str): Ruta a verificar.
        
        Returns:
            str: Mensaje de error o validación.
        """
        if not os.path.exists(path):
           return error_message(f'La ruta "{path}" no existe.')
        else:
            return (f'  ● Verificación de existencia de la ruta "{path}": ✔.')
    
    def check_folder(self, path:str) -> str:
        """
        Verifica que la ruta en cuestión exista.
        Args:
            path (str): Ruta a verificar.
        
        Returns:
            str: Mensaje de error o validación.
        """
        if not os.path.isdir(path):
            return error_message(f'La ruta "{path}" debe ser una carpeta.')
        else:
            return (f'  ● Verificación de existencia de la carpeta "{path}": ✔.')
    
    def check_file_tipe(self, files:Union[list[str], tuple[str]]) -> str:
        """
        Verifica que los archivos dentro de una lista son de tipo imágen de formato tif, jpg, jpeg, png, gif o bmp.

        Args:
            files (Union[list[str], tuple[str]]): Iterable (lista o tupla) con los nombres de los archivos junto con su extensión.
        
        Returns:
            str: Mensaje de error o validación.
        """
        extensions = ['.tif', '.jpg', '.jpeg', '.png', '.gif', '.bmp']
        if not files:
            return error_message(f'No hay archivos dentro de la lista.')
        for file in files:
            ext = file[-4:].lower()
            if ext not in extensions:
                return error_message(f'Se detectaron archivos con un formato diferente a tif, jpg, jpeg, png, gif o bmp. Especificamente "{file}".')
        return (f'  ● Verificación de los tipos de archivos como imágenes: ✔.')

class VerifyWarnings():
    def __init__(self) -> None:
        """
        Contiene métodos que verifican diferentes condiciones recomendadas sobre las variables de entrada. Se pausa 
        temporalmente el flujo de ejecución para notificar al usuario en caso de no validar satisfactoriamente la 
        verificación.
        """
        pass

    def check_limits(self, var:Union[int, float], l_min:Union[int, float], l_max:Union[int, float], label:str) -> str:
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
        if not l_min <= var <= l_max:
            return warning_message(f'Se recomienda que la variable "{label}" tenga valores entre {l_min} y {l_max}. No es obligatorio para la ejecución del algoritmo, pero puede afectar en los resultados del entrenamiento.')
        else:
            return  (f'  ● Verificación de límites para la variable "{label}": ✔.')