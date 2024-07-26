from keras._tf_keras.keras import layers, optimizers, models, regularizers
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping, History
from keras._tf_keras.keras.models import load_model
from resources.verify_variables import VerifyErrors as ve, VerifyWarnings as vw
from resources.message import method_menssage
import random
from typing import Union
import numpy as np
import json
from resources.general import create_path_save

class CreateFullAuto:
    """
    Crea y guarda la arquitectura la arquitectura compilada y sin entrenar del full auto encoder, en un archivo h5,
    según la estructura de este trabajo.
    """
    def __init__(self) -> None:
        self.full_auto      = models.Sequential(name='full_autoencoder')

    def check_create_model(self, verify_errors:Union[str,None]=None, verify_warnings:Union[str,None]=None, kernels:int=None, dim:int=None, number_layers:int=None, mode_l1:Union[str,None]=None, mode_l2:Union[str,None]=None, param_l1:float=None, param_l2:float=None, mode_do:Union[str,None]=None, param_do:float=None, lr:float=None) -> str:
        """
        Aplica verificaciones en los argumentos que recibe la función, deteniendo el flujo de ejecución en caso de error,
        o enviando un mensaje temporal a la consola en caso de advertencia.

        Args:
            verify_errors   (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            veify_warnings  (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.   
            kernels         (int): Número de kernels con el que se crea la capa inicial.
            dim             (int): Dimensión m de los datos de entrada (m,m,3)
            number_layers   (int): Número de capas del encoder.
            mode_l1         (Union[str,None]): Modo de uso de regularización l1.
                - 'all':    Todas las capas tendrán regularización l1.
                - 'random': Capas elegidas aleatoriamente tendrán regularización l1.
                - None:     Ninguna capa tendrá regularización l1.
            mode_l2         (Union[str,None]): Modo de uso de regularización l2.
                - 'all':    Todas las capas tendrán regularización l2.
                - 'random': Capas elegidas aleatoriamente tendrán regularización l2.
                - None:     Ninguna capa tendrá regularización l1.
            param_l1        (float): Valor de regularización l1.
            param_l2        (float): Valor de regularización l2.
            mode_do         (Union[str,None]): Modo de uso de drop out.
                - 'all':    Todas las capas tendrán drop out.
                - 'random': Capas elegidas aleatoriamente tendrán drop out.
                - None:     Ninguna capa tendrá drop out.
            lr              (float): Valor de learning rate.

        Returns:
            str: Confirmación de validación.
        """
        # Determina si ejecuta o no la verificación de errores
        if verify_errors == 'y':    
            method_menssage(self.check_create_model.__name__, 'Verifica los posibles errores al ingresar los argumentos de la función create_model')
            # Verifica los posibles valores que puede recibir como argumento una variable
            modes =  ['all', 'random', None]
            ve().check_arguments(mode_l1, modes, 'modo de uso de regularización l1')
            ve().check_arguments(mode_l2, modes, 'modo de uso de regularización l2')
            ve().check_arguments(mode_do, modes, 'modo de uso de drop out')
            # Verifica el tipo de dato
            ve().check_type(kernels, int, 'número inicial de kernels')
            ve().check_type(dim, int, 'dimensión de los datos de entrada')
            ve().check_type(number_layers, int, 'número de capas en el encoder')
            ve().check_type(param_l1, float, 'valor de regularización l1')
            ve().check_type(param_l2, float, 'valor de regularización l2')
            ve().check_type(param_do, float, 'valor de drop out')
            ve().check_type(lr, float, 'taza de aprendizaje')
            # Verifica las variables que deben ser números positivos
            ve().check_positive(kernels, 'número inicial de kernels')
            ve().check_positive(param_l1, 'valor de regularización l1')
            ve().check_positive(param_l2, 'valor de regularización l2')
            ve().check_positive(param_do, 'valor de drop out')
            ve().check_positive(lr, 'taza de aprendizaje')
            # Verifica las variables que deben ser números pares
            ve().check_par(kernels, 'número inicial de kernels')
            # Verifica la relación entre la variable dim y number_layers
            ve().check_dim_layers(dim, number_layers)
        elif verify_errors == 'n' or verify_errors == None:
            print('No se hará validación de errores a los argumentos de la función, esto puede suscitar errores.')
        else:
            ve().check_arguments(verify_errors, ['y', 'n', None], 'validación de errores en argumentos')
        # Determina si ejecuta o no la verificación de advertencias
        if verify_warnings == 'y':    
            method_menssage(self.check_create_model.__name__, 'Verifica las posibles advertencias al ingresar los argumentos de la función create_model')
            # Varifica que los argumentos numéricos estén dentro de límites esperados
            vw().check_limits(kernels, 8, 32, 'número inicial de kernels')
            vw().check_limits(param_l1, 0.00001, 0.1, 'valor de regularización l1')
            vw().check_limits(param_l2, 0.00001, 0.1, 'valor de regularización l2')
            vw().check_limits(param_do, 0.1, 0.4, 'valor de drop out')
            vw().check_limits(lr, 0.0001, 0.1, 'taza de aprendizaje')
            return print(f'\nValidación completa.\n')
        elif verify_warnings == 'n' or verify_warnings == None:
            return print('No se hará validación de advertencias a los argumentos de la función, esto puede suscitar errores o desmejorar los resultados del entrenamiento.')
        else:
            return ve().check_arguments(verify_warnings, ['y', 'n', None], 'validación de adavertencias en argumentos')

    def check_train_model(self, verify_errors:Union[str,None]=None, dataset:np.ndarray=None, patience:int=None, epochs:int=None, batch_size:int=None, dim:int=None) -> str:
        """
        Aplica verificaciones en los argumentos que recibe la función, deteniendo el flujo de ejecución en caso de error.
 
        Args:
            verify_errors   (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            dataset         (np.ndarray): Conjunto de datos de entrenamiento.
            patience        (int): Epocas de espera para la parada temprana.
            epochs          (int): Epocas totales del entrenamiento.
            batch_size      (int): Tamaño del lote de datos para el entrenamiento.

        Returns:
            str: Confirmación de validación.   
        """
        # Determina si ejecuta o no la verificación de errores
        if verify_errors == 'y':    
            method_menssage(self.check_train_model.__name__, 'Verifica los posibles errores al ingresar los argumentos de la función train_model')
            # Verifica el tipo de dato
            ve().check_type(dataset, np.ndarray, 'dataset')
            ve().check_type(patience, int, 'epocas de espera para la parada temprana')
            ve().check_type(epochs, int, 'epocas de entrenamiento')
            ve().check_type(batch_size, int, 'tamaño de lote')
            # Verifica las variables que deben ser números positivos
            ve().check_positive(patience, 'epocas de espera para la parada temprana')
            ve().check_positive(epochs, 'epocas de entrenamiento')
            ve().check_positive(batch_size, 'tamaño de lote')
            # Verifica que el dataset tenga una dimensionalidad de arreglo esperada
            ve().check_dim_dataset(dataset, dim)
            return print(f'\nValidación completa.\n')
        elif verify_errors == 'n' or verify_errors == None:
            return print('No se hará validación de errores a los argumentos de la función, esto puede suscitar errores.')
        else:
            return ve().check_arguments(verify_errors, ['y', 'n', None], 'validación de errores en argumentos')

    def check_save_model(self, verify_errors:Union[str,None]=None, pth_save_model:str=None) -> str:
        """
        Aplica verificaciones en los argumentos que recibe la función, deteniendo el flujo de ejecución en caso de error.

        Args:
            verify_errors   (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            pth_save_model  (str): Ruta de la carpeta donde se guardará el modelo de la red neuronal.
        
        Returns:
            str: Confirmación de validación.
        """
        # Determina si ejecuta o no la verificación de errores
        if verify_errors == 'y':    
            method_menssage(self.check_save_model.__name__, 'Verifica los posibles errores al ingresar los argumentos de la función save_model')
            # Verifica el tipo de dato
            ve().check_type(pth_save_model, str, 'ruta de guardado para el modelo')
            # Verifica la existencia de las rutas
            ve().check_path(pth_save_model)
            # Verifica que las rutas sean carpetas
            ve().check_folder(pth_save_model)
            return print(f'\nValidación completa.\n')
        elif verify_errors == 'n' or verify_errors == None:
            return print('No se hará validación de errores a los argumentos de la función, esto puede suscitar errores.')
        else:
            return ve().check_arguments(verify_errors, ['y', 'n', None], 'validación de errores en argumentos')

    def check_save_history(self, verify_errors:Union[str,None]=None, pth_save_history:str=None) -> str:
        """
        Aplica verificaciones en los argumentos que recibe la función, deteniendo el flujo de ejecución en caso de error.
 
        Args:
            verify_errors   (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            pth_save_history(str): Ruta de la carpeta donde se guardará la información del entrenamiento.
        
        Returns:
            str: Confirmación de validación.
        """
        # Determina si ejecuta o no la verificación de errores
        if verify_errors == 'y':
            method_menssage(self.check_save_history.__name__, 'Verifica los posibles errores al ingresar los argumentos de la función save_history')
            # Verifica el tipo de dato
            ve().check_type(pth_save_history, str, 'ruta de guardado para los datos del entrenamiento')
            # Verifica la existencia de las rutas
            ve().check_path(pth_save_history)
            # Verifica que las rutas sean carpetas
            ve().check_folder(pth_save_history)
            return print(f'\nValidación completa.\n')
        elif verify_errors == 'n' or verify_errors == None:
            return print('No se hará validación de errores a los argumentos de la función, esto puede suscitar errores.')
        else:
            return ve().check_arguments(verify_errors, ['y', 'n', None], 'validación de errores en argumentos')

    def check_load_full_auto(self, verify_errors:Union[str,None]=None, pth_model:str=None) -> str:
        """
        Aplica verificaciones a los argumentos que recibe la función, deteniendo el flujo de ejecución en caso de error.

        Args:
            verify_errors   (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            pth_model       (str): Ruta de la carpeta de donde se cargará el modelo.
        
        Returns:
            str: Confirmación de validación.
        """
        # Determina si ejecuta o no la verificación de errores
        if verify_errors == 'y':
            method_menssage(self.check_save_history.__name__, 'Verifica los posibles errores al ingresar los argumentos de la función load_full_auto')
            # Verifica el tipo de dato
            ve().check_type(pth_model, str, 'ruta de guardado para los datos del entrenamiento')
            # Verifica la existencia de las rutas
            ve().check_path(pth_model)
            return print(f'\nValidación completa.\n')
        elif verify_errors == 'n' or verify_errors == None:
            return print('No se hará validación de errores a los argumentos de la función, esto puede suscitar errores.')
        else:
            return ve().check_arguments(verify_errors, ['y', 'n', None], 'validación de errores en argumentos')

    def addreg(self, param_l1, param_l2) -> regularizers:
        """
        Devuelve las instancias de regularización de keras l1, l2 y l1l1.

        Returns:
            regularizers: Objeto regularizador.
        """
        add_l1      = regularizers.L1(param_l1)                    # Define el regularizador l1 para la capa
        add_l2      = regularizers.L2(param_l2)                    # Define el regularizador l2 para la capa
        add_l1l2    = regularizers.L1L2(param_l2, param_l2)   # Define el regularizador l1 y l2 para la misma capa
    
        return add_l1, add_l2, add_l1l2
    
    def adddo(self, mode_do, param_do) -> Union[Dropout, None]:
        """
        Devuelve las capa con drop out, dado el modo de uso y el valor del parámetro.

        Returns:
            Dropout:    Capa con drop out.
            None:       No agrega la capa.
        """
        if mode_do == 'all' or mode_do == 'random':       # Evalua la condición que debe cumplir el valor de la variable mode_do para añadir drop out
            return self.full_auto.add(Dropout(rate=param_do))  # Define la capa con drop out
        else:
            return None
        
    def rand_bol(self) -> random:
        """
        Devuelve True o False de manera aleatoria.

        Returns:
            random: Valor True o False.
        """
        return random.choice([True, False]) # Objeto random con posibilidad de devolver True o False
    
    def choice_reg(self, mode_l1, mode_l2) -> None:
        """
        Añade los objetos de regularización a las capas, apoyandose de la función add_reg y condicionales para aplicar
        la regularización pertinente o no hacerlo.

        Esta función no espera argumentos ni devuelve valores.
        """
        if mode_l1 == 'all' and mode_l2 == 'all':
            _, _, reg = self.addreg()
        elif mode_l1 == 'all' and mode_l2 == None:
            reg, *_ = self.addreg()
        elif mode_l1 == None and mode_l2 == 'all':
            _, reg, _ = self.addreg()
        elif mode_l1 == 'random' and mode_l2 == 'random':
            l1 = self.rand_bol()
            l2 = self.rand_bol()
            if l1 == True and l2 == True:
                _, _, reg = self.addreg()
            elif l1 == False and l2 == True:
                _, reg, _ = self.addreg()
            elif l1 == True and l2 == False:
                reg, *_ = self.addreg()
            elif l1 == False and l2 == False:
                reg = None
        elif mode_l1 == 'random' and mode_l2 == None:
            l1 = self.rand_bol()
            if l1 == True:
                reg, *_ = self.addreg()
            else:
                reg = None
        elif mode_l1 == None and mode_l2 == 'random':
            l2 = self.rand_bol()
            if l2 == True:
                _, reg, _ = self.addreg()
            else:
                reg = None
        elif mode_l1 == 'random' and mode_l2 == 'all':
            l1 = self.rand_bol()
            if l1 == True:
                _, _, reg = self.addreg()
            else:
                _, reg, _ = self.addreg()
        elif mode_l1 == 'all' and mode_l2 == 'random':
            l2 = self.rand_bol()
            if l2 == True:
                _, _, reg = self.addreg()
            else:
                reg, *_ = self.addreg()

    def choice_do(self, mode_do):
        if mode_do == 'all':
            do = self.adddo()
        elif mode_do == 'random':
            choice = self.rand_bol()
            if choice == True:
                do = self.adddo()
            else:
                do = None
        elif mode_do == None:
            do = None

    def create_model(self, verify_errors:Union[str,None]=None, verify_warnings:Union[str,None]=None, kernels:int=None, dim:int=None, number_layers:int=None, mode_l1:Union[str,None]=None, mode_l2:Union[str,None]=None, param_l1:float=None, param_l2:float=None, mode_do:Union[str,None]=None, param_do:float=None, lr:float=None) -> models.Sequential:
        """
        Construye la arquitectura de la red dados los parámetros, guardandolos en el modelo "full_auto_encoder".

        Args:
            kernels         (int): Número de kernels con el que se crea la capa inicial.
            dim             (int): Dimensión m de los datos de entrada (m,m,3)
            number_layers   (int): Número de capas del encoder.
            mode_l1         (Union[str,None]): Modo de uso de regularización l1.
                - 'all':    Todas las capas tendrán regularización l1.
                - 'random': Capas elegidas aleatoriamente tendrán regularización l1.
                - None:     Ninguna capa tendrá regularización l1.
            mode_l2         (Union[str,None]): Modo de uso de regularización l2.
                - 'all':    Todas las capas tendrán regularización l2.
                - 'random': Capas elegidas aleatoriamente tendrán regularización l2.
                - None:     Ninguna capa tendrá regularización l1.
            param_l1        (float): Valor de regularización l1.
            param_l2        (float): Valor de regularización l2.
            mode_do         (Union[str,None]): Modo de uso de drop out.
                - 'all':    Todas las capas tendrán drop out.
                - 'random': Capas elegidas aleatoriamente tendrán drop out.
                - None:     Ninguna capa tendrá drop out.
            lr              (float): Valor de learning rate.

        Returns:
            models.Sequential: Objeto que contiene el modelo secuencial de ren neuronal.
        """
        ve().check_provided([kernels, dim, number_layers, lr], 'crear el modelo', self.create_model, 'Define la arquitectura y la almacena en un modelo secuencial')
        self.check_create_model(verify_errors, verify_warnings, kernels, dim, number_layers, mode_l1, mode_l2, param_l1, param_l2, mode_do, param_do, lr)
        for lay in range(1, number_layers+1):
            if lay == 1 :
                self.full_auto.add(layers.Input(shape=(dim,dim,3)))
                self.full_auto.add(layers.Conv2D(kernels, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg(mode_l1, mode_l2)))
                self.choice_do(mode_do)
                self.full_auto.add(layers.MaxPooling2D((2, 2)))
                kernels *= 2
            elif lay == number_layers:
                self.full_auto.add(layers.Conv2D(kernels, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg(mode_l1, mode_l2)))
                self.choice_do(mode_do)
                self.full_auto.add(layers.MaxPooling2D((2, 2)))
                kernels -= kernels//2
            elif not lay in [1, number_layers]:
                self.full_auto.add(layers.Conv2D(kernels, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg(mode_l1, mode_l2)))
                self.choice_do(mode_do)
                self.full_auto.add(layers.MaxPooling2D((2, 2)))
                kernels *= 2
        for lay in range(1, number_layers+1):
            if lay == number_layers:
                self.full_auto.add(layers.Conv2DTranspose(kernels, (2, 2), activation='relu', kernel_regularizer=self.choice_reg(mode_l1, mode_l2)))
                self.choice_do(mode_do)
                self.full_auto.add(layers.UpSampling2D((2,2)))
                self.full_auto.add(layers.Conv2DTranspose(3, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg(mode_l1, mode_l2)))
            else:
                self.full_auto.add(layers.Conv2DTranspose(kernels, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg(mode_l1, mode_l2)))
                self.choice_do(mode_do)
                self.full_auto.add(layers.UpSampling2D((2,2)))
            kernels -= kernels//2

        optimizer_autoencoder = optimizers.Adam(learning_rate=lr)                           # Define el optimizador
        self.full_auto.compile(optimizer=optimizer_autoencoder, loss='mse', metrics=['mse'])  # Compila el modelo

        return self.full_auto
    
    def train_model(self, verify_errors:Union[str,None]=None, dataset:np.ndarray=None, patience:int=None, epochs:int=None, batch_size:int=None, dim:int=None) -> Union[models.Sequential, History]:
        """
        Define el mecanismo de parada temprana, entrena el modelo y guarda los datos del entrenamiento.

        Returns:
            models.sequential:  Modelo entrenado.
            History:            Información del entrenamiento.
        """
        self.check_train_model(verify_errors, dataset, patience, epochs, batch_size, dim)
        ve().check_provided([dataset, patience, epochs, batch_size, dim], 'entrenar el modelo', self.train_model, 'Entrena el modelo y devuelve el historial')
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        self.history = self.full_auto.fit(dataset, dataset, epochs=epochs, batch_size=batch_size, shuffle=False, validation_split=0.20, verbose=0, callbacks=[early_stopping])
        return self.full_auto, self.history
    
    def save_model(self, verify_errors:Union[str,None]=None, model:models.Sequential=None, pth_save_model:str=None) -> str:
        """
        Guarda el modelo que se ecuentre almacenado en la variable full_auto en formato h5.

        Args:
            Verify_errors    (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            model           (models.Sequential): Modelo de la red neuronal.
            pth_save_model  (str): Ruta a la carpeta donde se guardará el modelo.
        
        Returns:
            str: Texto de confirmación del guardado.
        """
        ve().check_provided([model, pth_save_model], 'guardar el modelo', self.save_model, 'Guarda el modelo en un archivo keras')
        self.check_save_model(verify_errors, pth_save_model)
        pth_save = create_path_save(pth_save_model, 'full_auto_encoder', 'keras')   # Define la ruta donde se guardará el archivo
        model.save(pth_save)                                                        # Guarda el modelo
        return print(f'\nModelo guardado con éxito en "{pth_save}".\n')

    def save_history(self, verify_errors:Union[str,None]=None, pth_save_history:str=None) -> str:
        """
        Guarda la información del entrenamiento como un archivo json.

        Args:
            Verify_errors        (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            pth_save_history    (str): Ruta a la carpeta donde se guardará el modelo.
        
        Returns:
            str: Texto de confirmación del guardado.
        """
        ve().check_provided([pth_save_history],'guardar el historial de entrenamiento', self.save_history, 'Guarda el historial de entrenamiento en un archivo json')
        self.check_save_history(verify_errors, pth_save_history)
        history_dict = self.history.history
        pth_save = create_path_save(pth_save_history, 'train_history', 'json') # Define la ruta donde se guardará el archivo
        with open(pth_save, 'w') as file:                                           # Abre el archivo json
            json.dump(history_dict, file)                                           # Guarda el archivo
        return print(f'\nHistorial de entrenamiento guardado con éxito en "{pth_save}".\n')

    def load_full_auto(self, verify_errors:Union[str,None]=None, pth_model:str=None) -> str:
        """
        Carga un modelo desde un archivo en formato keras.

        Args:
            Verify_errors        (str): Indica si se verifican o no los valores de los argumentos.
                - 'y':  Si se realiza el proceso de verificación.
                - 'n':  No se realiza el proceso de verificación.
                - None: No se realiza el proceso de verificación.
            pth_load            (str): Ruta a la carpeta de donde se cargará el modelo.
        
        Returns:
            (models.Sequential): Modelo de la red neuronal.
        """
        ve().check_provided([pth_model], 'cargar el modelo', self.load_full_auto, 'Carga el modelo desde un archivo keras')
        self.check_load_full_auto(verify_errors, pth_model)
        model = load_model(pth_model)
        return model