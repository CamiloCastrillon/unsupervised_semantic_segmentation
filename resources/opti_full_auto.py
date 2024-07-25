import os
import optuna
import numpy as np
from resources.message import method_menssage
from typing import Union
from resources.create_architecture import CreateFullAuto as cfa

class OptimizeFullAuto:
    """
    Crea el objeto de estudio con optuna para optimizar el full auto encoder y permite consultar los parámetros obtenidos
    """
    def __init__(self, dataset:np.ndarray=None, nl_min:int=None, nl_max:int=None, dim:int=None, lr_min:float=None, lr_max:float=None, ep_min:int=None, ep_max:int=None, ba_min:int=None, ba_max:int=None, ink:int=None, mode_l1:Union[str, None]=None, rl1_min:float=None, rl1_max:float=None, mode_l2:Union[str, None]=None, rl2_min:float=None, rl2_max:float=None, mode_do:Union[str, None]=None, do_min:float=None, do_max:float=None, esp_min:int=None, esp_max:int=None) -> None:
        """
        Inicializa los argumentos de la clase.
        
        Args:
            dataset (np.ndarray): Dataset.Conjunto de datos de entrenamiento.
            nl_min  (int): Minimum number of layers. Número mínimo de capas en el encoder.
            nl_max  (int): Maximum number of layers. Número máximo de capas en el encoder.
            dim     (int): Dimension. Valor la dimensión m de los datos de entrada (mxmx3).
            lr_min  (float): Minimum Learning Rate. Taza de aprendizaje mínima.
            lr_max  (float): Maximum Learning Rate. Taza de aprendizaje máxima.
            ep_min  (int): Minimum number of epochs. Número de épocas de mínimas de entrenamiento.
            ep_max  (int): Maximum number of epochs. Número de épocas de máximas de entrenamiento.
            ba_min  (int): Minimum Batch Size. Tamaño de lote mínimo.
            ba_max  (int): Maximum Batch Size. Tamaño de lote máximo.
            ink     (int): Initial number of kernels: Número de kernels para iniciar el full autoencoder
            mode_l1 (Union[str,None]): Modo de uso de regularización l1.
                - 'all':    Todas las capas tendrán regularización l1.
                - 'random': Capas elegidas aleatoriamente tendrán regularización l1.
                - None:     Ninguna capa tendrá regularización l1.
            rl1_min (float): Minimum L1 regularization. Regularización L1 mínima.
            rl1_max (float): Maximum L1 regularization. Regularización L1 máxima.
            mode_l2 (Union[str,None]): Modo de uso de regularización l2.
                - 'all':    Todas las capas tendrán regularización l2.
                - 'random': Capas elegidas aleatoriamente tendrán regularización l2.
                - None:     Ninguna capa tendrá regularización l2.
            rl2_min (float): Minimum L2 regularization. Regularización L2 mínima.
            rl2_max (float): Maximum L2 regularization. Regularización L2 máxima.
            mode_do (Union[str,None]): Modo de uso de drop out.
                - 'all':    Todas las capas tendrán drop out.
                - 'random': Capas elegidas aleatoriamente tendrán drop out.
                - None:     Ninguna capa tendrá drop out.
            do_min  (float): Minumum DropOut. Porcentaje de apagado de neuronas mínimo.
            do_max  (float): Maximum DropOut. Porcentaje de apagado de neuronas máximo.
            esp_min (int): Minimum Early Stopping Patience. Número mínimo de epocas de espera para la para temprana.
            esp_max (int): Maximum Early Stopping Patience. Número máximo de epocas de espera para la para temprana.
        """
        self.dataset    = dataset
        self.nl_min     = nl_min
        self.nl_max     = nl_max
        self.dim        = dim
        self.lr_min     = lr_min
        self.lr_max     = lr_max
        self.ep_min     = ep_min
        self.ep_max     = ep_max 
        self.ba_min     = ba_min
        self.ba_max     = ba_max 
        self.ink        = ink
        self.mode_l1    = mode_l1
        self.rl1_min    = rl1_min
        self.rl1_max    = rl1_max
        self.mode_l2    = mode_l2
        self.rl2_min    = rl2_min
        self.rl2_max    = rl2_max
        self.mode_do    = mode_do
        self.do_min     = do_min
        self.do_max     = do_max
        self.esp_min    = esp_min
        self.esp_max    = esp_max

    def optuna_study(self, trial):
        """
        Crea el objeto de estudio de optuna.
        """
        method_menssage(self.optuna_study.__name__, 'Crea el objeto de estudio de optuna')
        # Parámetros a variar de la arquitectura.
        nl  = trial.suggest_int('number_layers', self.nl_min, self.nl_max)
        l1  = trial.suggest_float('l1_regularization', self.rl1_min, self.rl1_max)
        l2  = trial.suggest_float('l2_regularization', self.rl2_min, self.rl2_max)
        do  = trial.suggest_float('drop_out', self.do_min, self.do_max)
        lr  = trial.suggest_float('learning_rate', self.nl_min, self.nl_max)
        # Parámetros a variar en el entrenamiento.
        esp = trial.suggest_int('early_stopping', self.esp_min, self.esp_max)
        ep  = trial.suggest_int('epochs', self.ep_min, self.ep_max) 
        bs  = trial.suggest_int('batch_size', self.ba_min, self.ba_max)
        # Crea la arquitectura
        model = cfa().create_model('y', 'y', self.ink, self.dim, nl, self.mode_l1, self.mode_l2, l1, l2, self.mode_do, do, lr)
        # Entrena la arquitectuira
        _, history00 = cfa().train_model('y', self.dataset, esp, ep, bs, self.dim)
        # Obtiene la pérdida del entrenamiento
        val_loss = history00.history['val_loss'][-1]
        return val_loss
    
    def execute_study(self):
        study = optuna.create_study(direction='minimize')   # Crea el estudio con el objetivo de minimizar la pérdida del modelo.
        study.optimize(self.optuna_study, n_trials=5)       # Ejecuta el estudio.

        return print(f'\nMejores parametros encontrados: {study.best_params}\nPéridida más baja en la configuración optima: {study.best_value}')