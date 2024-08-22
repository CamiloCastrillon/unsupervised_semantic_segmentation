import pickle
import numpy as np

path_calib_params   = 'C:/camilo/calibrate_camera/xiaomi_redmi_note_11.pkl'

def load_calibratrion(path_calib_params):
    # Cargar los datos desde el archivo .pkl
    with open(path_calib_params, 'rb') as file:
        calibration_data = pickle.load(file)

    # Extraer los coeficientes de distorsión
    camera_matrix   = calibration_data['camera_matrix']

    # Convertir a formato ndarray
    camera_matrix   = np.array(camera_matrix)

    Fx={camera_matrix[0][0]}   
    Cx={camera_matrix[0][2]} 
    Fy={camera_matrix[1][1]}
    Cy={camera_matrix[1][2]}

    text_matrix = f'{Fx};0;{Cx};0;{Fy};{Cy};0;0;1'
    