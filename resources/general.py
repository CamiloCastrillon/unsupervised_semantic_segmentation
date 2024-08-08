import os
from datetime import datetime

def create_path_save(folder:str, name:str, ext:str):
    """
    Crea las rutas de guardados de los archivos, tiendo la ruta de la carpeta donde se va a guardar,
    el nombre y la extensión del archivo con el formato: [//ruta_a_la_carpeta/nombre_fecha_hora.extensión].

    Returns:
        str: La ruta del archivo a guardar.
    """
    # Genera el nombre único para el dataset
    fecha_hora_actual       = datetime.now()                                    # Obtiene la fecha y hora actual
    fecha_hora_formateada   = fecha_hora_actual.strftime("%d/%m/%Y %I:%M %p")   # Expresa la fecha y hora en un formato diferente
    fecha_hora_formateada   = fecha_hora_formateada.replace(" ", "_")           # Reemplaza los espacios por guiones bajos
    fecha_hora_formateada   = fecha_hora_formateada.replace("/", "-")           # Reemplaza los guines medios por slash
    fecha_hora_formateada   = fecha_hora_formateada.replace(":", "-")           # Reemplaza los dos puntos por guiones medios
    name = f'{name}_{fecha_hora_formateada}.{ext}'                              # Define el nombre del archivo a guardar
    pth_save = os.path.join(folder, name)                                     # Crea la ruta del archivo a guardar
    return pth_save