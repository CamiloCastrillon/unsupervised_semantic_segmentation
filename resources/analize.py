import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras._tf_keras.keras.models import load_model
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from resources.general import create_path_save
from resources.create_dataset import GenDataAutoencoder as gda

class AnalizeFullAuto:
    def __init__(self) -> None:
        """
        Contiene los métodos para analizar los resultados del entrenamiento y predicción del full auto encoder.
        """
        pass
    
    def plot_histories(self, path_history:str, path_save:str) -> str:
        """
        Grafica los datos de pérdida y mse, del historial de entrenamiento desde una archivo json, muestra la gráfica y la guarda.
        
        Args:
            path_history    (str): Ruta con el archivo json del historial.
            path_save       (str): Ruta a la carpeta donde se desea guardar la imágen.
        
        Returns:
            str: Texto de confirmación del proceso finalizado.
        """
        with open(path_history, 'r') as f:
            history_dict = json.load(f)
            
        loss        = history_dict['loss']
        val_loss    = history_dict['val_loss']
        mse         = history_dict['mse']
        val_mse     = history_dict['val_mse']
        epochs      = range(1, len(loss) + 1)
        
        loss_min    = min(loss)
        loss_max    = max(loss)
        val_loss_min= min(val_loss)
        val_loss_max= max(val_loss)
        mse_min     = min(mse)
        mse_max     = max(mse)
        val_mse_min = min(val_mse)
        val_mse_max = max(val_mse)
        num_epochs  = len(loss)

        loss_text = f'Número de épocas: {num_epochs}\nPérdida del entrenamiento\n   ●Mínima: {loss_min}\n   ●Máxima: {loss_max}\nPérdida de validación\n   ●Mínima: {val_loss_min}\n   ●Máxima: {val_loss_max}'
        mse_text  = f'Número de épocas: {num_epochs}\nMSE del entrenamiento\n   ●Mínimo: {mse_min}\n   ●Máximo: {mse_max}\nMSE de validación\n   ●Mínimo: {val_mse_min}\n   ●Máximo: {val_mse_max}'

        plt.figure(figsize=(16, 6))
        plt.rcParams['font.family'] = 'Times New Roman'

        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, color='#6F65FF', label='Pérdida de entrenamiento')
        plt.plot(epochs, val_loss, color='#D34A4A', label='Pérdida de validación')
        plt.title('Pérdida Durante el Entrenamiento', fontdict={'weight': 'bold', 'size': 12})
        plt.xlabel('Épocas', fontdict={'weight': 'bold', 'size': 12})
        plt.ylabel('Pérdida', fontdict={'weight': 'bold', 'size': 12})
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1, color='#CAC9C9')

        # Ajustar el espaciado para permitir espacio para el texto
        plt.subplots_adjust(bottom=0.3)  # Ajustar este valor si es necesario

        # Añadir texto con valores máximo y mínimo
        plt.text(0, -0.15, loss_text, 
                ha='left', va='top', transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

        plt.subplot(1, 2, 2)
        plt.plot(epochs, mse, color='#6F65FF', label='MSE de entrenamiento')
        plt.plot(epochs, val_mse, color='#D34A4A', label='MSE de validación')
        plt.title('Error Médio Cuadrático Durante el Entrenamiento', fontdict={'weight': 'bold', 'size': 12})
        plt.xlabel('Épocas', fontdict={'weight': 'bold', 'size': 12})
        plt.ylabel('MSE', fontdict={'weight': 'bold', 'size': 12})
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1, color='#CAC9C9')

        # Añadir texto con valores máximo y mínimo
        plt.text(0, -0.15, mse_text, 
                ha='left', va='top', transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.5'))

        entire_path_save = create_path_save(path_save, 'history', 'jpg')
        plt.savefig(entire_path_save, dpi=300, bbox_inches='tight')

        plt.show()
        
        return print('Figura guardada con éxito.')
    
    def one_predict(self, path_img:str, dim:int, path_model:str, path_save_predicts:str, path_save_image:str) -> np.ndarray:
        """
        Crea las predicciones del full auto encoder para una imágen y las guarda en un archivo npy.
        
        Args:
            path_img            (str): Ruta de la imágen de la cual se quieren generar predicciones.
            dim                 (int): Dimensión de las secciones que se van a evaluar.
            path_model          (str): Ruta del modelo con el cual se generarán las predicciones.
            path_save_predicts  (str): Ruta de la carpeta donde se guardarán las predicciones.
            path_save_image     (str): Ruta de la carpeta donde se guardará la imagen comparativa.
            
        Returns:
            np.ndarray: Array de numpy con las predicciones
        """
        
        full_auto   = load_model(path_model)
        img         =  cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2RGB)
        w, h        = img.shape[1], img.shape[0]
        
        secciones = gda().make_data(img, w, h, dim)
        secciones_array = np.array(secciones)
        predicciones = full_auto.predict(secciones_array)
        
        entire_path_save_predicts = create_path_save(path_save_predicts, 'predicciones', 'npy')
        np.save(entire_path_save_predicts, predicciones)

        large_image = np.zeros((h, w, 3), dtype=secciones_array.dtype)
        index = 0
        for row in range (0, h, dim):
            for col in range(0, w, dim):
                large_image[row:row+dim, col:col+dim] = predicciones[index]
                index += 1
        
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, axis = plt.subplots(1, 2, figsize=(8, 6))

        # Muestra la imagen original
        axis[0].imshow(img)
        axis[0].set_title('Imagen Original', fontdict={'weight': 'bold', 'size': 12})
        axis[0].axis('off')

        # Muestra la imagen reconstruida
        axis[1].imshow(large_image)
        axis[1].set_title('Imagen Reconstruida', fontdict={'weight': 'bold', 'size': 12})
        axis[1].axis('off')
        
        entire_path_save_image = create_path_save(path_save_image, 'predicciones', 'jpg')
        plt.savefig(entire_path_save_image, dpi=300, bbox_inches='tight')
        plt.show()
        return print('Implementación del modelo realizada con éxito')