import os
import optuna
import numpy as np
from resources.message import error_message, warning_message, method_menssage

class OptimizeFullAuto:
    """
    Crea el objeto de estudio con optuna para optimizar el full auto encoder y permite consultar los parámetros obtenidos
    """
    def __init__(self, dataset, nl_min, nl_max, dim, lr_min, lr_max, ep_min, ep_max, ba_min, ba_max, ink, rl1_min, rl1_max, rl2_min, rl2_max, do_min, do_max, esp_min, esp_max):
        self.dataset    = dataset   # Dataset: Conjunto de datos de entrenamiento
        self.nl_min     = nl_min    # Minimum number of layers: Número mínimo de capas en el encoder
        self.nl_max     = nl_max    # Maximum number of layers: Número máximo de capas en el encoder
        self.dim        = dim       # Dimension: Valor la dimensión m de los datos de entrada (mxmx3)
        self.lr_min     = lr_min    # Minimum Learning Rate: Taza de aprendizaje mínima
        self.lr_max     = lr_max    # Maximum Learning Rate: Taza de aprendizaje máxima
        self.ep_min     = ep_min    # Minimum number of epochs: Número de épocas de mínimas de entrenamiento
        self.ep_max     = ep_max    # Maximum number of epochs: Número de épocas de máximas de entrenamiento
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

    def check_type(self, var, type, label):
        if not isinstance(var, type):
            error_message(f'la variable "{label}" debe ser de tipo {type.__name__}.')

    def check_numeric_min_max(self, var_min, var_max, var_type, label_min, label_max):
        self.check_type(var_min, var_type, label_min)
        self.check_type(var_max, var_type, label_max)
        if var_min >= var_max:
            error_message(f'La variable "{label_min}" debe ser menor al número máximo.')
        elif var_min <= 0:
            error_message(f'La variable "{label_min}" debe ser mayor a cero.')

    def check_limits(self, var, l_min, l_max, label):
        if not l_min <= var <= l_max:
            warning_message(f'Se recomienda que la variable "{label}" tenga valores entre {l_min} y {l_max}. No es obligatorio para la ejecución del algoritmo, pero puede afectar en los resultados del entrenamiento.')

    def check_values(self):
        """
        Evalua los posibles errores al ingresar los argumentos de la clase
        """
        method_menssage(self.check_values.__name__)
        # Evalua el tipo de la variable dataset
        self.check_type(self.dataset, np.ndarray, 'dataset')
        if not self.dataset.ndim == 5:
            error_message('Las dimensiones de la variable dataset no son las cinco esperadas.')

        # Evalua las variables nl_min y nl_max
        self.check_numeric_min_max(self.nl_min, self.nl_max, int, 'número de capas mínimo', 'número de capas máximo')

        # Evalua la variable dim
        self.check_type(self.dim, int, 'dimensión de los datos de entrada')

        # Evalua la dimensión de los datos de entrada y si corresponde al número de capas máximo
        cheack_dim = self.dim
        for i in range(self.nl_max):
            cheack_dim /= 2
            if round(cheack_dim) < 3:         
                error_message('El encoder tiene una cantidad máxima de capas superior a la esperada por la dimensión de los datos de entrenamiento.')

        # Evalua la variable lr_min y lr_max
        self.check_numeric_min_max(self.lr_min, self.lr_max, float, 'learning rate mínimo', 'learning rate máximo')

        # Evalua si los valores de lr_min y lr_max son recomendados
        self.check_limits(self.lr_min, 0.0001, 0.001, 'learning rate mínimo')
        self.check_limits(self.lr_max, 0.01, 0.1, 'learning rate máximo')
        
        # Evalua la variable ep_min y ep_max
        self.check_numeric_min_max(self.ep_min, self.ep_max, int, 'número de epocas de entrenamiento mínimas', 'número de epocas de entrenamiento mínimas')

        # Evalua la variable ba_min y ba_max
        self.check_numeric_min_max(self.ba_min, self.ba_max, int, 'batch size mínimo', 'batch size máximo')

        # Evalua que el ba_max no sea mayor o igual al número de datos por muestra del dataset
        if self.ba_max >= self.dataset.shape[1]:
            error_message('El batch size máximo no puede ser mayor al número de datos por muestra del dataset.')
        
        # Evalua la variable ink
        self.check_type(self.ink, int, 'número inicial de kernels')
        if self.ink <= 0:
            error_message('La variable del número inicial de kernels debe ser un número mayor a cero.')
        elif not self.ink % 2 == 0:
            error_message('La variable del número incial de kernels debe ser un número par.')
        self.check_limits(self.ink, 8, 32, 'número inicial de kernels')

        # Evalua la variable rl1_min y rl1_max
        self.check_numeric_min_max(self.rl1_min, self.rl1_max, float, 'regularización l1 mínima', 'regularización l1 máxima')
        self.check_limits(self.rl1_min, 0.00001, 0.0001, 'regularización l1 mínima')
        self.check_limits(self.rl1_max, 0.001, 0.1, 'regularización l1 máxima')

        # Evalua la variable rl2_min y rl2_max
        self.check_numeric_min_max(self.rl2_min, self.rl2_max, float, 'regularización l2 mínima', 'regularización l2 máxima')
        self.check_limits(self.rl2_min, 0.00001, 0.0001, 'regularización l2 mínima')
        self.check_limits(self.rl2_max, 0.001, 0.1, 'regularización l2 máxima')

        # Evalua las variables do_min y do_max
        self.check_numeric_min_max(self.do_min, self.do_max, float, 'drop out mínimo', 'drop out máximo')
        self.check_limits(self.do_min, 0.1, 0.2, 'drop out mínimo')
        self.check_limits(self.do_max, 0.2, 0.4, 'drop out máximo')

        # Evalua las variables esp_min y esp_max
        self.check_numeric_min_max(self.esp_min, self.esp_max, int, 'número mínimo de epocas de espera para la para temprana', 'número máximo de epocas de espera para la para temprana')
        self.check_limits(self.esp_min, 1, 10, 'número mínimo de epocas de espera para la para temprana')
        self.check_limits(self.esp_max, 10, 20, 'número máximo de epocas de espera para la para temprana')

    def optuna_study(self, trial):
        """
        Crea el objeto de estudio de optuna
        """
        method_menssage(self.optuna_study.__name__)
        self.check_values()
        
        number_layers   = trial.suggest_int('number_layers', self.nl_min, self.nl_max)
        learning_rate   = trial.suggest_float('learning_rate', self.nl_min, self.nl_max)
        epochs          = trial.suggest_int('epochs', self.ep_min, self.ep_max) 
        batch_size      = trial.suggest_int('batch_size', self.ba_min, self.ba_max)
        kernels         = self.ink
