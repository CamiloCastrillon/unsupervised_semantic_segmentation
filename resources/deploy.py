import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras._tf_keras.keras.models import load_model
from resources.general import create_path_save
from resources.create_architecture import CreateClassifier as cc
from sklearn.cluster import KMeans
import numpy as np


def predict_sequential(path_model:str, path_dataset:str, path_save_predicts:str) -> str:
    """
    Genera las predicciones de un modelo secuencial (full auto encoder o encoder) y las guarda en formato npy.
    
    Args:
        path_model          (str): Ruta con el archivo del modelo keras.
        path_dataset        (str): Ruta con el archivo del dataset npy.
        path_save_predicts  (str): Ruta de la carpeta donde se desea guardar la predicción.
        
    Returns:
        str: Texto de confirmación del guardado.
    """
    dataset                     = np.load(path_dataset)
    model                       = load_model(path_model)
    model_predicts              = model.predict(dataset)
    entire_path_save_predicts   = create_path_save(path_save_predicts, 'predicts', 'npy')
    np.save(entire_path_save_predicts, model_predicts)
    return print('Predicciones realizadas y guardadas con éxito')

def predict_som(path_sompy:str, path_dataset:str, n_classes:int, path_save_predicts:str) -> str:
    """
    Implementa el mapa para obtener la matriz bmu en un arreglo, y la clasificación de los pesos correspondientes a la misma usando kmeans, almacenandolo
    en otro arreglo, guardando ambos en un arreglo que contiene a ambos. Guarda todo en un archivo npy.
    
    Args:
        path_sompy          (str): Ruta del archivo pkl que contiene el som entrenado.
        path_dataset        (str): Ruta al dataset de caracteristicas a evaluar en el som.
        n_classes           (int): Número de clases que se desean distinguir del som.
        path_save_predicts  (str): Ruta de la carpeta donde se desean guardar las predicciones del som.
    
    Returns:
        str: Texto de confirmación del guardado.
    """
    dataset = np.load(path_dataset)
    sompy   = cc().load_sompy(path_sompy)

    bmu             = sompy.project_data(dataset)
    neuron_weights  = sompy.codebook.matrix

    bmu_weights = []
    for winning_neuron in bmu:
        weight = neuron_weights[winning_neuron]
        bmu_weights.append(weight)
        
    bmu_weights = np.array(bmu_weights)

    kmeans = KMeans(n_clusters=n_classes, random_state=0)
    kmeans.fit(bmu_weights)

    labels = kmeans.labels_
    predicts_array = np.array([bmu,labels])
    entire_path_save_predicts = create_path_save(path_save_predicts, 'som', 'npy')
    np.save(entire_path_save_predicts, predicts_array)
    
    return print('Predicciones realizadas con éxito')