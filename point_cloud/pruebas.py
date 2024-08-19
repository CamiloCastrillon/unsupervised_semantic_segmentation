import cv2
import numpy as np
import open3d as o3d
import os 
import numpy as np

folder_images       = 'C:/camilo/imgs/aguacate/'
focal_length_mm     = 26
sensor_width_mm     = (1/2.74)*25.4
sensor_height_mm    = (1/2.74)*25.4
image_width_px      = 1836/2
image_height_px     = 4080/2

def get_paths(folder_images):
    """
    Obtiene las imágenes de una carpeta y añade las rutas a una lista.
    """
    print('   🟢 Obteniendo las rutas de las imágenes.')
    path_images     = []
    files_images    = os.listdir(folder_images)
    for file in files_images:
        path = folder_images+file
        path_images.append(path)
    return path_images

def pre_process_image(path_images):
    """
    Carga las imágenes, las convierte a escala de grises y las guarda
    """
    print('   🟢 Pre-procesando imagenes.')
    images_gray = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY) for path in path_images]
    return images_gray

def get_features(images_gray):
    """
    Itera sobre las imágenes en escala de grises para obtener sus puntos clave y descriptores.
    """
    print('   🟢 Obteniendo características.')
    sift        = cv2.SIFT_create(nfeatures=10000)
    key_points  = []
    descriptors = []

    for imagen_gris in images_gray:
        kp, des = sift.detectAndCompute(imagen_gris, None)
        key_points.append(kp)
        descriptors.append(des)
    return key_points, descriptors

def match_features(descriptors):
    """
    Empareja las caracteristicas entre las imágenes.
    """
    print('   🟢 Emparejando características.')
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = []

    for i in range(len(descriptors) - 1):
        matches.append(flann.knnMatch(descriptors[i], descriptors[i + 1], k=2))
    return matches

def filter_matches(matches):
    """
    Filtra los emparejamientos.
    """
    print('   🟢 Filtrando características.')
    # Filtrar emparejamientos usando el ratio test de Lowe
    good_matches = []
    for match in matches:
        buenos = []
        for m, n in match:
            if m.distance < 0.7 * n.distance:
                buenos.append(m)
        good_matches.append(buenos)
    return good_matches

def calculate_3d_position(good_matches, key_points):
    """
    Calcula las coordenadas 3d de cada punto.
    """
    print('   🟢 Calculando puntos.')
    
    # Obtener puntos emparejados
    points1 = []
    points2 = []
    for i in range(len(good_matches)):
        points1.append(np.float32([key_points[i][m.queryIdx].pt for m in good_matches[i]]))
        points2.append(np.float32([key_points[i + 1][m.trainIdx].pt for m in good_matches[i]]))
    
    # Estimar la matriz fundamental usando RANSAC
    F, mask = cv2.findFundamentalMat(points1[0], points2[0], cv2.FM_RANSAC)
    
    # Supongamos que la distancia focal es conocida
    focal_length_mm     = 26  # Distancia focal en milímetros (ejemplo)
    # Suposiciones razonables para el tamaño del sensor y la resolución de la imagen
    sensor_width_mm     = (1/2.74)*25.4  # Ancho del sensor en milímetros (ejemplo)
    sensor_height_mm    = (1/2.74)*25.4  # Altura del sensor en milímetros (ejemplo)
    image_width_px      = 1836/2  # Ancho de la imagen en píxeles (ejemplo)
    image_height_px     = 4080/2  # Altura de la imagen en píxeles (ejemplo)

    # Convertir la distancia focal a píxeles
    f_x = (focal_length_mm * image_width_px) / sensor_width_mm
    f_y = (focal_length_mm * image_height_px) / sensor_height_mm

    # Suponer que el punto principal está en el centro de la imagen
    c_x = image_width_px / 2
    c_y = image_height_px / 2

    # Matriz intrínseca de la cámara
    K = np.array([[f_x, 0, c_x],
                [0, f_y, c_y],
                [0, 0, 1]])

    # Calcular la matriz esencial
    E = K.T @ F @ K
    
    # Descomponer la matriz esencial para obtener las matrices de rotación y traslación
    _, R, t, _ = cv2.recoverPose(E, points1[0], points2[0], K)

    # Matrices de proyección de las cámaras
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    P1 = K @ P1
    P2 = K @ P2
    
    # Triangular puntos para obtener coordenadas 3D
    points_4d = cv2.triangulatePoints(P1, P2, points1[0].T, points2[0].T)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)
    return points_3d

def create_point_cloud(points_3d):
    """
    Crea la nube de puntos.
    """
    print('   🟢 Creando la nube de puntos.')
    # Convertir puntos 3D a un formato adecuado
    points = np.vstack(points_3d).reshape(-1, 3)
    # Crear la nube de puntos
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def visualize(point_cloud):
    """
    Grafica la nube de puntos.
    """
    print('   🟢 Cargando la gráfica.')
    # Visualizar la nube de puntos
    o3d.visualization.draw_geometries([point_cloud])

def results(arg, label):
    print('\n🔽 Resultados:')
    print(f'   ✅ {label}: {type(arg)}')
    print(f'   ✅ {label}: {len(arg)}')

def work_flow():
    """
    Ejecuta todas las funciones del flujo de trabajo
    """
    print('\n🔽 Ejecutando el flujo de trabajo:')
    path_images             = get_paths(folder_images)
    images_gray             = pre_process_image(path_images)
    key_points, descriptors = get_features(images_gray)
    matches                 = match_features(descriptors)
    good_matches            = filter_matches(matches)
    points_3d               = calculate_3d_position(good_matches, key_points)
    point_cloud             = create_point_cloud(points_3d)
    visualize(point_cloud)
        
def main():
    try:
        work_flow()
    except Exception as e:
        print(f'\n⛔ Error:\n{e}\n')

main()