import pickle

# Variables'
path_save_calibration   = 'C:/camilo/calibrate_camera/xiaomi_redmi_note_11.pkl'

# Cargar los datos desde el archivo .pkl
with open(path_save_calibration, 'rb') as file:
    calibration_data = pickle.load(file)

# Extraer los coeficientes de distorsión
camera_matrix   = calibration_data['camera_matrix']
dist_coeffs     = calibration_data['dist_coeffs']
rvects          = calibration_data['rvecs']
tvecs           = calibration_data['tvecs']

# Mostrar los coeficientes de distorsión
print(f'\nMatriz de la camara:\n{camera_matrix}\n\nCoeficientes:\n   Fx={camera_matrix[0][0]}\n   Cx={camera_matrix[0][2]}\n   Fy={camera_matrix[1][1]}\n   Cy={camera_matrix[1][2]}')
