import open3d as o3d
import os 
import numpy as np
import cv2

path_ply = 'C:/camilo/imgs/pts.ply'    
# Rutas de las imágenes y parámetros de cámara (esto debe ser configurado adecuadamente)
folder_imgs         = 'C:/camilo/imgs/aguacate_kmeans/'
files_imgs          =  os.listdir(folder_imgs)
class_images = []
for file in files_imgs:
    path_img = folder_imgs+file
    class_images.append(path_img)

def check_file(file_path):
    if not os.path.exists(file_path):
        print(f"El archivo no existe: {file_path}")
        return False
    
    # Verificar la extensión del archivo
    valid_extensions = ['.ply', '.xyz', '.obj']
    if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
        print(f"Formato de archivo no soportado: {file_path}")
        return False
    
    # Verificar que el archivo no esté vacío
    if os.path.getsize(file_path) == 0:
        print(f"El archivo está vacío: {file_path}")
        return False
    
    return True

def load_and_visualize_point_cloud(file_path):
    if not check_file(file_path):
        return
    
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(pcd.points) == 0:
            print(f"La nube de puntos está vacía: {file_path}")
            return
        
        o3d.visualization.draw_geometries([pcd], window_name="Nube de Puntos 3D")
    except Exception as e:
        print(f"Error al cargar la nube de puntos: {e}")

def assign_classes_to_point_cloud(ply_file, class_images):
    # Cargar la nube de puntos desde el archivo PLY
    pcd = o3d.io.read_point_cloud(ply_file)
    
    # Leer las imágenes de clase
    class_images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in class_images]

    # Obtener el número de puntos y dimensiones de las imágenes
    num_points = len(pcd.points)
    image_height, image_width = class_images[0].shape
    
    # Crear un array para los colores de los puntos
    colors = np.zeros((num_points, 3))
    
    # Suponiendo que las imágenes y los puntos están en el mismo orden
    for i, img in enumerate(class_images):
        # Suponemos que la nube de puntos y las imágenes están alineadas
        # Esto puede requerir una proyección adecuada en casos más complejos
        img_class = img.flatten()
        # Aquí se requiere lógica adicional para mapear los puntos a las clases
        # Ejemplo simplificado:
        for idx, point in enumerate(pcd.points):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < image_width and 0 <= y < image_height:
                class_id = img_class[y * image_width + x]
                # Mapear clase a color
                color = [1, 0, 0] if class_id == 0 else [0, 1, 0] if class_id == 1 else [0, 0, 1] if class_id == 2 else [0,0,0]
                colors[idx] = color
    
    # Asignar colores a la nube de puntos
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualizar la nube de puntos
    o3d.visualization.draw_geometries([pcd], window_name="Nube de Puntos con Clases")

assign_classes_to_point_cloud(path_ply, class_images)