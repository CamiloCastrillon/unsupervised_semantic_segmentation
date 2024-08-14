import os
# Rutas de las imágenes y parámetros de cámara (esto debe ser configurado adecuadamente)
folder_imgs         = 'C:/camilo/imgs/aguacate_kmeans/'
files_imgs          =  os.listdir(folder_imgs)
image_paths = []
for file in files_imgs:
    path_img = folder_imgs+file
    image_paths.append(path_img)
    
print(image_paths)
