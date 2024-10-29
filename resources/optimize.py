import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo mostrar advertencias y errores
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import optuna
import numpy as np
from typing import Union
from resources.create_architecture import CreateFullAuto as cfa
from resources.general import create_path_save
from resources.verify_variables import VerifyErrors as ve
from resources.create_architecture import CreateClassifier as cc
from sompy.sompy import SOMFactory

class OptimizeFullAuto:
    """
    Crea el objeto de estudio con optuna para optimizar el full auto encoder y permite consultar los parámetros obtenidos
    """
    def __init__(self, dataset:np.ndarray=None, nl_min:int=None, nl_max:int=None, dim:int=None, lr_min:float=None, lr_max:float=None, ep_min:int=None, ep_max:int=None, ba_min:int=None, ba_max:int=None, ink:list[int]=None, mode_l1:Union[str, None]=None, rl1_min:float=None, rl1_max:float=None, mode_l2:Union[str, None]=None, rl2_min:float=None, rl2_max:float=None, mode_do:Union[str, None]=None, do_min:float=None, do_max:float=None, esp_min:int=None, esp_max:int=None, n_trials:int=None, pth_save_params:str=None, umbral:float=None, pth_save_model:str=None, pth_save_hist:str=None) -> None:
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
            ink_min         (list[int]): Initial number of kernels: Número de kernels para iniciar el full autoencoder.
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
        ink = trial.suggest_categorical('kernels', self.ink)
        # Crea la arquitectura
        model = cfa().create_model('n', 'n', ink, self.dim, nl, self.mode_l1, self.mode_l2, l1, l2, self.mode_do, do)
        # Entrena la arquitectuira
        trained_model, history00 = cfa().train_model('n', model, self.dataset, esp, ep, bs, self.dim, lr)
        # Obtiene la pérdida del entrenamiento
        val_loss = history00.history['val_loss'][-1]
        # Si el modelo ensayado en el intento correspondiente tiene una perdida igual o menor al umbral, el modelo se guarda.
        if val_loss <= self.umbral:
            cfa().save_model('n', trained_model, f'full_auto_encoder_trained_loss{val_loss}',self.pth_save_model)
            cfa().save_history('n', history00, self.pth_save_hist)
            print(f'Se guardó un modelo con pérdida: {val_loss}')
        return val_loss

    def save_params(self) -> str:
        """
        Guarda en un txt los mejores parámetros y resultados del estudio de optuna.

        Returns:
            str: Texto de confirmación de guardado del archivo.
        """
        ve().check_provided([self.pth_save_params], 'guardar en un txt los mejores parámetros y resultados del estudio de optuna', self.save_params, 'Guarda en un txt los mejores parámetros y resultados del estudio de optuna')
        # Crea el texto
        text = f'Mejor valor de pérdida encontrado: {self.best_value}\n\nMejores parámetros para este estudio:\n'
        for clave, valor in self.best_params.items():
            add = f'    {clave}={valor}\n'
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
        print(f'\nMejores parametros encontrados: {study.best_params}\nPéridida más baja en la configuración optima: {study.best_value}')
        self.best_params    = study.best_params
        self.best_value     = study.best_value
        self.save_params()
        return print('Ejecución del estudio finalizada.')

class OptimizeSOM:
    def __init__(self, hilos:int, path_dataset:str, mapsize:list[int], mask:any, mapshape:list[str], lattice:list[str], normalization:list[str], initialization:list[str], neighborhood :list[str], training :list[str], name:str, component_names:list[str], train_rough_len_min:int, train_rough_len_max:int, train_finetune_len_min:int, train_finetune_len_max:int, n_trials:int, umbral:float, path_save_params:str, path_save_models:str) -> None:
        
        self.hilos                  = hilos
        self.dataset                = np.load(path_dataset)
        self.mapsize                = mapsize
        self.mask                   = mask
        self.mapshape               = mapshape
        self.lattice                = lattice
        self.normalization          = normalization
        self.initialization         = initialization
        self.neigborhood            = neighborhood
        self.training               = training
        self.name                   = name
        self.component_names        = component_names
        self.train_rough_len_min    = train_rough_len_min
        self.train_rough_len_max    = train_rough_len_max
        self.train_finetune_len_min = train_finetune_len_min
        self.train_finetune_len_max = train_finetune_len_max
        self.n_trials               = n_trials
        self.umbral                 = umbral
        self.path_save_params       = path_save_params
        self.path_save_models       = path_save_models

    def create_study(self, trial) -> float:
        
        opt_mapshape            = trial.suggest_categorical('mapshape',self.mapshape)
        opt_lattice             = trial.suggest_categorical('laticce', self.lattice)
        opt_normalization       = trial.suggest_categorical('normalization', self.normalization)
        opt_inizialization      = trial.suggest_categorical('initialization', self.initialization)
        opt_neigborhood         = trial.suggest_categorical('neighborhood', self.neigborhood)
        opt_training            = trial.suggest_categorical('training', self.training)
        opt_train_rough_len     = trial.suggest_int('train_rough_len', self.train_rough_len_min, self.train_rough_len_max)
        opt_train_finetune_len  = trial.suggest_int('train_finetune_len', self.train_finetune_len_min, self.train_finetune_len_max)

        sompy = SOMFactory().build(data     = self.dataset, 
                           mapsize          = self.mapsize, 
                           mask             = self.mask, 
                           mapshape         = opt_mapshape, 
                           lattice          = opt_lattice, 
                           normalization    = opt_normalization, 
                           initialization   = opt_inizialization, 
                           neighborhood     = opt_neigborhood, 
                           training         = opt_training, 
                           name             = self.name, 
                           component_names  = self.component_names)

        sompy.train(n_job=self.hilos, verbose=False, train_rough_len=opt_train_rough_len, train_finetune_len=opt_train_finetune_len)

        topographic_error = sompy.calculate_topographic_error()
        if topographic_error <= self.umbral:
            cc().save_sompy(sompy, f'sompy_trained_{self.mapsize}_te{topographic_error}', self.path_save_models)
            print(f'Modelo guardado con error topográfico: {topographic_error}')
        return topographic_error

    def save_params(self) -> str:
        text = f'Mejor valor de error topográfico encontrado: {self.best_value}\n\nMejores parámetros para este estudio:\n'
        for clave, valor in self.best_params.items():
            add = f'    {clave}={valor}\n'
            text = text+add
        # Crea la ruta de guardado 
        entire_path_save_params = create_path_save(self.path_save_params, 'param_opti', 'txt')
        # Guarda el archivo txt
        with open(entire_path_save_params, "w", encoding="utf-8") as archivo:
            archivo.write(text)
        return print(f'\nInformación del mejor intento del estudio guardada en:"{entire_path_save_params}".')

    def execute_study(self) -> str:
        study = optuna.create_study(direction='minimize')
        study.optimize(self.create_study, n_trials=self.n_trials)
        print(f'\nMejores parametros encontrados: {study.best_params}\nPéridida más baja en la configuración optima: {study.best_value}')
        self.best_params    = study.best_params
        self.best_value     = study.best_value
        self.save_params()
        return print('Ejecución del estudio finalizada.')