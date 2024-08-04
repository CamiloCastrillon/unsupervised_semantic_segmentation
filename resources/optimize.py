import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo mostrar advertencias y errores
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import optuna
import numpy as np
from typing import Union
from resources.create_architecture import CreateFullAuto as cfa
from resources.general import create_path_save
from resources.verify_variables import VerifyErrors as ve, VerifyWarnings as vw

class OptimizeFullAuto:
    """
    Crea el objeto de estudio con optuna para optimizar el full auto encoder y permite consultar los parámetros obtenidos
    """
    def __init__(self, dataset:np.ndarray=None, nl_min:int=None, nl_max:int=None, dim:int=None, lr_min:float=None, lr_max:float=None, ep_min:int=None, ep_max:int=None, ba_min:int=None, ba_max:int=None, ink:int=None, mode_l1:Union[str, None]=None, rl1_min:float=None, rl1_max:float=None, mode_l2:Union[str, None]=None, rl2_min:float=None, rl2_max:float=None, mode_do:Union[str, None]=None, do_min:float=None, do_max:float=None, esp_min:int=None, esp_max:int=None, n_trials:int=None, save_param:Union[str,None]=None, pth_save_params:str=None, umbral:float=None, pth_save_model:str=None, pth_save_hist:str=None) -> None:
        """
        Inicializa los argumentos de la clase.
        
        Args:
            dataset         (np.ndarray): Dataset.Conjunto de datos de entrenamiento.
            nl_min          (int): Minimum number of layers. Número mínimo de capas en el encoder.
            nl_max          (int): Maximum number of layers. Número máximo de capas en el encoder.
            dim             (int): Dimension. Valor la dimensión m de los datos de entrada (mxmx3).
            lr_min          (float): Minimum Learning Rate. Taza de aprendizaje mínima.
            lr_max          (float): Maximum Learning Rate. Taza de aprendizaje máxima.
            ep_min          (int): Minimum number of epochs. Número de épocas de mínimas de entrenamiento.
            ep_max          (int): Maximum number of epochs. Número de épocas de máximas de entrenamiento.
            ba_min          (int): Minimum Batch Size. Tamaño de lote mínimo.
            ba_max          (int): Maximum Batch Size. Tamaño de lote máximo.
            ink             (int): Initial number of kernels: Número de kernels para iniciar el full autoencoder
            mode_l1         (Union[str,None]): Modo de uso de regularización l1.
                - 'all':    Todas las capas tendrán regularización l1.
                - 'random': Capas elegidas aleatoriamente tendrán regularización l1.
                - None:     Ninguna capa tendrá regularización l1.
            rl1_min         (float): Minimum L1 regularization. Regularización L1 mínima.
            rl1_max         (float): Maximum L1 regularization. Regularización L1 máxima.
            mode_l2         (Union[str,None]): Modo de uso de regularización l2.
                - 'all':    Todas las capas tendrán regularización l2.
                - 'random': Capas elegidas aleatoriamente tendrán regularización l2.
                - None:     Ninguna capa tendrá regularización l2.
            rl2_min         (float): Minimum L2 regularization. Regularización L2 mínima.
            rl2_max         (float): Maximum L2 regularization. Regularización L2 máxima.
            mode_do         (Union[str,None]): Modo de uso de drop out.
                - 'all':    Todas las capas tendrán drop out.
                - 'random': Capas elegidas aleatoriamente tendrán drop out.
                - None:     Ninguna capa tendrá drop out.
            do_min          (float): Minumum DropOut. Porcentaje de apagado de neuronas mínimo.
            do_max          (float): Maximum DropOut. Porcentaje de apagado de neuronas máximo.
            esp_min         (int): Minimum Early Stopping Patience. Número mínimo de epocas de espera para la para temprana.
            esp_max         (int): Maximum Early Stopping Patience. Número máximo de epocas de espera para la para temprana.
            n_trials        (int): Número de intentos en la optimización de hiper parámetros.
            save_param      (Union[str,None]): Determina si se quiere o no guardar el archivo txt con los mejores parametros encontrados en el estudio.
                - 'y':  Se guardan.
                - 'n':  No se guardan.
                - None: No se guardan.
            pth_save_params (str): Ruta de la carpeta de guardado para el archivo txt con los mejores parámetros.
            umbral          (float): Umbral de pérdida con el que se decide si se conserva el modelo o no.
            pth_save_model  (str): Ruta de la carpeta de guardado del modelo.
            pth_save_hist   (str): Ruta de la carpeta de guardado guardado del historial de entrenamiento.
        """
        self.dataset        = dataset
        self.nl_min         = nl_min
        self.nl_max         = nl_max
        self.dim            = dim
        self.lr_min         = lr_min
        self.lr_max         = lr_max
        self.ep_min         = ep_min
        self.ep_max         = ep_max 
        self.ba_min         = ba_min
        self.ba_max         = ba_max 
        self.ink            = ink
        self.mode_l1        = mode_l1
        self.rl1_min        = rl1_min
        self.rl1_max        = rl1_max
        self.mode_l2        = mode_l2
        self.rl2_min        = rl2_min
        self.rl2_max        = rl2_max
        self.mode_do        = mode_do
        self.do_min         = do_min
        self.do_max         = do_max
        self.esp_min        = esp_min
        self.esp_max        = esp_max
        self.n_trials       = n_trials
        self.save_param     = save_param
        self.pth_save_params= pth_save_params
        self.umbral         = umbral
        self.pth_save_model = pth_save_model
        self.pth_save_hist  = pth_save_hist

    def create_study(self, trial) -> float:
        """
        Crea el objeto de estudio de optuna.

        Args:
            verify_errors   (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            veify_warnings  (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            trial           None: Este argumento es necesario para el funcionamiento de optuna, no se debe llenar.

        Returns:
            float: Pérdida del modelo entrenado.
        """
        ve().check_provided([self.umbral, self.pth_save_model, self.pth_save_hist], 'crear el objeto de estudio con optuna', self.create_study, 'Crea el objeto de estudio de optuna')
        # Parámetros a variar de la arquitectura.
        nl  = trial.suggest_int('number_layers', self.nl_min, self.nl_max)
        if not (self.rl1_min == None or self.rl1_max == None):
            l1  = trial.suggest_float('l1_regularization', self.rl1_min, self.rl1_max)
        else:
            l1 = None
        if not (self.rl2_min == None or self.rl2_max == None):
            l2  = trial.suggest_float('l2_regularization', self.rl2_min, self.rl2_max)
        else:
            l2 = None
        if not (self.do_min == None or self.do_max == None):
            do  = trial.suggest_float('drop_out', self.do_min, self.do_max)
        else:
            do = None
        lr  = trial.suggest_float('learning_rate', self.lr_min, self.lr_max)
        # Parámetros a variar en el entrenamiento.
        esp = trial.suggest_int('early_stopping', self.esp_min, self.esp_max)
        ep  = trial.suggest_int('epochs', self.ep_min, self.ep_max) 
        bs  = trial.suggest_int('batch_size', self.ba_min, self.ba_max)
        # Crea la arquitectura
        model = cfa().create_model('n', 'n', self.ink, self.dim, nl, self.mode_l1, self.mode_l2, l1, l2, self.mode_do, do)
        # Entrena la arquitectuira
        trained_model, history00 = cfa().train_model('n', model, self.dataset, esp, ep, bs, self.dim, lr)
        # Obtiene la pérdida del entrenamiento
        val_loss = history00.history['val_loss'][-1]
        # Si el modelo ensayado en el intento correspondiente tiene una perdida igual o menor al umbral, el modelo se guarda.
        if val_loss <= self.umbral:
            cfa().save_model('y', trained_model, self.pth_save_model)
            cfa().save_history('y', self.pth_save_hist)
        return val_loss
    
    def save_params(self) -> str:
        """
        Guarda en un txt los mejores parámetros y resultados del estudio de optuna.

        Returns:
            str: Texto de confirmación de guardado del archivo.
        """
        ve().check_provided([self.pth_save_params], 'guardar en un txt los mejores parámetros y resultados del estudio de optuna', self.save_params, 'Guarda en un txt los mejores parámetros y resultados del estudio de optuna')
        # Crea el texto
        text = f'Mejor valor de pérdida encontrado: {self.best_value}\nMejores parámetros para este estudio:\n'
        for clave, valor in self.best_params.items():
            add = f'{clave}={valor}\n'
            text = text+add
        # Crea la ruta de guardado 
        pth = create_path_save(self.pth_save_params, 'param_opti', 'txt')
        # Guarda el archivo txt
        with open(pth, "w", encoding="utf-8") as archivo:
            archivo.write(text)
        return print(f'\nInformación del mejor intento del estudio guardada en:"{pth}".')

    def execute_study(self) -> str:
        """
        Ejecuta el estudio de optuna.

        Args:

        Returns:
            str: Texto con los mejores resultados del estudio.
        """
        ve().check_provided([self.n_trials], 'ejecutar el estudio de optuna', self.execute_study, 'Ejecuta el estudio de optuna')
        study = optuna.create_study(direction='minimize')   # Crea el estudio con el objetivo de minimizar la pérdida del modelo.
        study.optimize(self.create_study, n_trials=self.n_trials)       # Ejecuta el estudio.
        result              = f'\nMejores parametros encontrados: {study.best_params}\nPéridida más baja en la configuración optima: {study.best_value}'
        print(result)
        self.best_params    = study.best_params
        self.best_value     = study.best_value
        # Guarda los mejores parámetros
        if self.save_param == 'y':
            self.save_params()
        elif self.save_param == 'n' or self.save_param == None:
            self.save_params()
        else:
            ve().check_arguments(self.save_param, [str, None], 'Indicador de si se desean o no, guardar los mejores parámetros encontrados')
        return print('Ejecución del estudio finalizada.')