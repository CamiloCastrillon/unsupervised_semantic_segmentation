import os
extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif']
ruta = 'c:/camilo/data/img.tif'
ext = os.path.splitext(ruta)[1].lower()
if ext in extensions:
    text = (f'La ruta {ruta} no es un archivo de imagen.')
else:
    pass