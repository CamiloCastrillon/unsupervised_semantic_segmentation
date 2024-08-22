import cv2
import glob
import numpy as np
import pickle

# Variables
folder_imgs             = 'C:/camilo/calibrate_camera/imgs_calib/'
path_save_calibration   = 'C:/camilo/calibrate_camera/xiaomi_redmi_note_11.pkl'

# Configurar el tamaño del tablero de ajedrez
chessboard_size = (17, 11)  # Número de esquinas internas por fila y columna
square_size = 6  # Tamaño del cuadrado en unidades del mundo real (e.g., 1 cm)

# Prepara los puntos de objeto (coordenadas 3D) para el tablero de ajedrez
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arreglos para guardar puntos del objeto y puntos de la imagen en todas las imágenes
objpoints = []  # Puntos 3D en el espacio del mundo real
imgpoints = []  # Puntos 2D en el espacio de la imagen

# Cargar las imágenes de calibración
images = glob.glob(folder_imgs+'*.jpg')

# Detectar las esquinas del tablero de ajedrez
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encuentra las esquinas del tablero de ajedrez
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Si las esquinas son encontradas, agrega puntos de objeto y puntos de imagen
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Dibuja y muestra las esquinas
        cv2.namedWindow('Esquinas del Tablero', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Esquinas del Tablero', 1020, 459)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Esquinas del Tablero', img)
        cv2.waitKey(1000)

cv2.destroyAllWindows()

# Calibra la cámara
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Mostrar los resultados de la calibración
print('\nMatriz de la cámara:\n', camera_matrix)
print('\nCoeficientes de distorsión:\n', dist_coeffs)
print('\nVectores de rotación:\n', rvecs)
print('\nVectores de traslación:\n', tvecs)

# Guardar los parámetros de calibración en un archivo
calibration_data = {
    'camera_matrix': camera_matrix,
    'dist_coeffs': dist_coeffs,
    'rvecs': rvecs,
    'tvecs': tvecs
}

print('\nGuardando datos de calibración.')
with open(path_save_calibration, 'wb') as f:
    pickle.dump(calibration_data, f)

print('Parámetros de calibración guardados exitosamente.')

