import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def initialize_proejct_variables():
    path_folder_images  = 'C:/camilo/uss/imgs/aguacate/'
    path_folder_pc      = 'C:/camilo/uss/imgs/point_cloud/'
    path_calib_params   = 'C:/camilo/calibrate_camera/xiaomi_redmi_note_11.pkl'

    return path_folder_images, path_folder_pc, path_calib_params

def load_images(image_dir):
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]
    images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in image_files]
    return images, image_files

def load_calibratrion(path_calib_params):
    # Cargar los datos desde el archivo .pkl
    with open(path_calib_params, 'rb') as file:
        calibration_data = pickle.load(file)

    # Extraer los coeficientes de distorsión
    camera_matrix   = calibration_data['camera_matrix']
    dist_coeffs     = calibration_data['dist_coeffs']

    # Convertir a formato ndarray
    camera_matrix   = np.array(camera_matrix)
    # Cambia el orden k1, k2, k3, p1, p2 a k1, k2, p1, p2, k3
    dist_coeffs     = np.array([dist_coeffs[0][0],dist_coeffs[0][1],dist_coeffs[0][3],dist_coeffs[0][4], dist_coeffs[0][2]])
    return camera_matrix, dist_coeffs

def detect_and_describe_features(image_files):
    orb = cv2.ORB_create()
    keypoints_list = []
    descriptors_list = []

    for file in image_files:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)

    return keypoints_list, descriptors_list

def match_features(descriptors_list):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_list = []

    for i in range(len(descriptors_list) - 1):
        matches = bf.match(descriptors_list[i], descriptors_list[i + 1])
        matches_list.append(matches)

    return matches_list

def find_pose(keypoints_list, matches_list, K):
    poses = []
    for i in range(len(matches_list)):
        pts1 = np.float32([keypoints_list[i][m.queryIdx].pt for m in matches_list[i]])
        pts2 = np.float32([keypoints_list[i + 1][m.trainIdx].pt for m in matches_list[i]])

        E, _ = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        poses.append((R, t))
    return poses

def rectify_images(img1, img2, K, dist_coeffs, R, t):
    h, w = img1.shape[:2]
    R1, R2, P1, P2, Q, _ = cv2.stereoRectify(K, dist_coeffs, K, dist_coeffs, (w, h), R, t)
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist_coeffs, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist_coeffs, R2, P2, (w, h), cv2.CV_32FC1)

    img1_rectified = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

    return img1_rectified, img2_rectified, Q

def compute_disparity(img1, img2):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(img1, img2)
    return disparity

def reconstruct_3d(disparity, Q):
    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    return points_3D

# Procesar todas las imágenes
def process_images(images, keypoints_list, matches_list, K, dist_coeffs, poses):
    all_points_3D = []
    
    for i in range(len(matches_list)):
        img1, img2 = images[i], images[i + 1]
        R, t = poses[i]

        img1_rectified, img2_rectified, Q = rectify_images(img1, img2, K, dist_coeffs, R, t)
        disparity = compute_disparity(img1_rectified, img2_rectified)
        points_3D = reconstruct_3d(disparity, Q)
        
        all_points_3D.append(points_3D)
    
    return all_points_3D

def fuse_point_clouds(all_points_3D):
    # Fusionar nubes de puntos en una sola nube de puntos.
    # Aquí se asume que todas las nubes de puntos están en el mismo sistema de coordenadas.
    combined_points_3D = np.vstack(all_points_3D)
    return combined_points_3D

def visualize_point_cloud(points_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = points_3D[:, :, 0].ravel()
    y = points_3D[:, :, 1].ravel()
    z = points_3D[:, :, 2].ravel()

    ax.scatter(x, y, z, c='r', marker='o')
    plt.show()

def save_point_cloud(points_3D, output_file):
    np.savetxt(output_file, points_3D.reshape(-1, 3), delimiter=' ')

def workflow():
    # Inicializa las variables del proyecto
    print('   🟢 Inizializando rutas.')
    path_folder_images, path_folder_pc, path_calib_params = initialize_proejct_variables()
    # Carga las imagenes
    print('   🟢 Cargando imagenes.')
    images, image_files = load_images(path_folder_images)
    # Obtiene la matriz de calibración y los coeficientes de distorción
    print('   🟢 Obteniendo datos de calibración.')
    camera_matrix, dist_coeffs = load_calibratrion(path_calib_params)
    # Detecta las caracteristicas de cada imagen
    print('   🟢 Detectando características.')
    keypoints_list, descriptors_list = detect_and_describe_features(image_files)
    # Empareja las caracteristicas que coinciden entre las imágenes
    print('   🟢 Emparejando caracteristicas.')
    matches_list = match_features(descriptors_list)
    # Encuentra la posición de las camaras
    print('   🟢 Estimando la posición de las camaras.')
    poses = find_pose(keypoints_list, matches_list, camera_matrix)
    # Calcula los puntos 3d
    print('   🟢 Calculando las coordenadas xyz.')
    all_points_3D = process_images(images, keypoints_list, matches_list, camera_matrix, dist_coeffs, poses)
    # Fusiona las nubes de puntos
    print('   🟢 Fusionando nubes de puntos.')
    combined_points_3D = fuse_point_clouds(all_points_3D)
    # Grafica la nube de puntos
    print('   🟢 Graficando la nube de puntos.')
    visualize_point_cloud(combined_points_3D)
    # Guarda la nube de puntos
    print('   🟢 Guardando la nube de puntos.')
    save_point_cloud(combined_points_3D, f'{path_folder_pc}combined_point_cloud.xyz')

def main():
    """
    try:
        workflow()
    except Exception as e:
        print(f'\n⛔ Error:\n{e}\n')
    """
    workflow()

# Ejecuta el flujo de trabajo
main()