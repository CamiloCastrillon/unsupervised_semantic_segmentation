from PIL import Image, ImageFilter
import os 
import re

folder_images_metadata  = 'C:/camilo/imgs/aguacate/'
folder_images           = 'C:/camilo/imgs/colored/'
folder_save             = 'C:/camilo/imgs/colored_metadata/'

# Obtiene las rutas con las imagenes con la metadata
path_images_metadata     = []
files_images_metadata    = os.listdir(folder_images_metadata)
for file in files_images_metadata:
    path = folder_images_metadata+file
    path_images_metadata.append(path)

# Obtiene la ruta de las imágenes a guardar con la metadata
path_images     = []
files_images    = os.listdir(folder_images)
for file in files_images:
    path = folder_images+file
    path_images.append(path)

# Función para extraer el número de la ruta de la imagen
def extraer_numero(ruta):
    match = re.search(r'(\d+)', ruta)
    return int(match.group(1)) if match else 0

# Ordenar la lista de rutas de imágenes
path_images_metadata = sorted(path_images_metadata, key=extraer_numero)
path_images = sorted(path_images, key=extraer_numero)

# Crea una lista con los metadatos de cada imagen
metadata = []
for path in path_images_metadata:
    img = Image.open(path)
    exif_data = img.info.get('exif')
    metadata.append(exif_data)

index_metadata = 0
for path in path_images:
    img = Image.open(path)
    path_save = folder_save+f'colored_metadata_{index_metadata+1}.jpg'
    img.save(path_save, format="JPEG", exif=metadata[index_metadata])
    index_metadata += 1
    print(f'    ->Imagen guardada en: "{path_save}".')