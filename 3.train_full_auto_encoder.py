from resources.create_architecture import CreateFullAuto as cfa
import numpy as np
"""
model       (models.Sequential): Objeto que contiene el modelo secuencial de ren neuronal.
dataset     (np.ndarray): Arreglo de numpy con los datos de entrenamiento.
patience    (int): Número de epocas de espera para la para temprana.
epochs      (int): Número de epocas para el entrenamiento.
batch_size  (int): Tamaño del lote.
dim         (int): Dimensión de los datos de entrada.
lr          (float): Valor de learning rate.
"""
cfa = cfa()
# Rutas con los datos a usar
pth_model   = 'C:/camilo/uss/models/full_auto_encoder/full_auto_encoder_1.keras'
pth_dataset = 'C:/camilo/uss/datasets/full_auto_encoder/dataset_dim50.npy'
# Rutas de guardado
pth_save_trained = 'C:/camilo/uss/models/full_auto_encoder/'
pth_save_history = 'C:/camilo/uss/histories/full_auto_encoder/'
# Parámetros del modelo
model       = cfa.load_any_model('y', pth_model)
dataset     = np.load(pth_dataset)
patience    = 10
epochs      = 144
batch_size  = 109
dim         = 50
lr          = 0.0009634736927387456
# Entrena el modelo y obtiene tanto el modelo como el historial
trained_model, history = cfa.train_model('y', model, dataset, patience, epochs, batch_size, dim, lr)
# Guarda el modelo
cfa.save_model('y', trained_model, 'full_auto_encoder_trained_1', pth_save_trained)
# Guarda el historial
cfa.save_history('y', pth_save_history)