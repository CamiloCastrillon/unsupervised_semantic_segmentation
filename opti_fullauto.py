import os
import optuna
import tensorflow as tf
from keras._tf_keras.keras import layers, optimizers, initializers, models, regularizers
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping

class OptimizeFullAuto:
    """
    Crea el objeto de estudio con optuna para optimizar el full auto encoder y permite consultar los parámetros obtenidos
    """
    
    def __init__(self, nl_min, nl_max, lr_min, lr_max, ep_min, ep_max, ba_min, ba_max, ink, rl1_min, rl1_max, rl2_min, rl2_max, do_min, do_max, esp_min, esp_max):
        
        self.nl_min     = nl_min    # Minimum number of layers: Número mínimo de capas totales del autoencoder
        self.nl_max     = nl_max    # Maximum number of layers: Número máximo de capas en el autoencoder
        self.lr_min     = lr_min    # Minimum Learning Rate: Taza de aprendizaje mínima
        self.lr_max     = lr_max    # Maximum Learning Rate: Taza de aprendizaje máxima
        self.ep_min     = ep_min    # Minimum number of epochs: Número de épocas de mínimas de entrnamiento
        self.ep_max     = ep_max    # Maximum number of epochs: Número de épocas de máximas de entrnamiento
        self.ba_min     = ba_min    # Minimum Batch Size: Tamaño de lote mínimo
        self.ba_max     = ba_max    # Maximum Batch Size: Tamaño de lote máximo
        self.ink        = ink       # Initial number of kernels: Número de kernels para iniciar el full autoencoder
        self.rl1_min    = rl1_min   # Minimum L1 regularization: Regularización L1 mínima
        self.rl1_max    = rl1_max   # Maximum L1 regularization: Regularización L1 máxima
        self.rl2_min    = rl2_min   # Minimum L2 regularization: Regularización L2 mínima
        self.rl2_max    = rl2_max   # Maximum L2 regularization: Regularización L2 máxima
        self.do_min     = do_min    # Minumum DropOut: Porcentaje de apagado de neuronas mínimo
        self.do_max     = do_max    # Maximum DropOut:Porcentaje de apagado de neuronas máximo
        self.esp_min    = esp_min   # Minimum Early Stopping Patience: Número mínimo de epocas de espera para la para temprana
        self.esp_max    = esp_max   # Maximum Early Stopping Patience: Número máximo de epocas de espera para la para temprana
    
    def check_values(self):
        pass
    #
    
    def optuna_study(self, trial):
        """
        Crea el objeto de estudio de optuna
        """
        
        self.check_values()
        
        number_layers   = trial.suggest_int('number_layers', self.nl_min, self.nl_max)
        learning_rate   = trial.suggest_float('learning_rate', self.nl_min, self.nl_max)
        epochs          = trial.suggest_int('epochs', self.ep_min, self.ep_max) 
        batch_size      = trial.suggest_int('batch_size', self.ba_min, self.ba_max)
        kernels         = self.ink
        
        full_auto = models.Sequential(name='full_autoencoder')