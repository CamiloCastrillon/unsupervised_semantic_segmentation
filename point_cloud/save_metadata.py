from PIL import Image, ImageFilter

path_img = 'C:/camilo/imgs/DJI_0935.jpg'
path_save = 'C:/camilo/imgs/DJI_0935_mod_metadata.jpg'

# Cargar la imagen con Pillow
with Image.open(path_img) as img:
 # Leer los metadatos EXIF
    exif_data = img.info.get('exif')

    # Aplicar una modificación a la imagen (ejemplo: aplicar un filtro de desenfoque)
    modified_image = img.filter(ImageFilter.GaussianBlur(radius=5))

    # Guardar la imagen modificada con los metadatos EXIF
    if exif_data:
        modified_image.save(path_save, format="JPEG", exif=exif_data)

    else:
        # Si no hay datos EXIF, simplemente guarda la imagen
        modified_image.save(path_save)