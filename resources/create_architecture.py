from keras._tf_keras.keras import layers, optimizers, models, regularizers
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping, History
from resources.verify_variables import VerifyErrors as ve, VerifyWarnings as vw
from resources.message import method_menssage
import random
from typing import Union
import numpy as np
import json
from general import create_path_save

class CreateFullAuto:
    """
    Crea y guarda la arquitectura la arquitectura compilada y sin entrenar del full auto encoder, en un archivo h5,
    según la estrcutura de este trabajo.

    Args:
        verify          (str): Indica si se verifican o no los valores de los argumentos.
            - 'y':  Si se realiza el proceso de verificación.
            - 'n':  No se realiza el proceso de verificación.
        kernels         (int): Número de kernels con el que se crea la capa inicial.
        dim             (int): Dimensión m de los datos de entrada (m,m,3)
        number_layers   (int): Número de capas del encader.
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
        dataset         (np.ndarray): Conjunto de datos de entrenamiento.
        patience        (int): Epocas de espera para la parada temprana.
        epochs          (int): Epocas totales del entrenamiento.
        batch_size      (int): Tamaño del lote de datos para el entrenamiento.
        pth_save_model  (str): Ruta de la carpeta donde se guardará el modelo de la red neuronal.
        pth_save_history(str): Ruta de la carpeta donde se guardará la información del entrenamiento.
    
    Returns:
        None: No se espera argumento de salida.
    """
    def __init__(self, verify:str, kernels:int, dim:int, number_layers:int, mode_l1:Union[str,None], mode_l2:Union[str,None], param_l1:float, param_l2:float, mode_do:Union[str,None], param_do:float, lr:float, dataset:np.ndarray, patience:int, epochs:int, batch_size:int, pth_save_model:str, pth_save_history:str) -> None:
        # Argumentos para definir el modelo
        self.full_auto      = models.Sequential(name='full_autoencoder')
        self.verify         = verify
        self.kernels        = kernels
        self.dim            = dim
        self.number_layers  = number_layers
        self.mode_l1        = mode_l1
        self.mode_l2        = mode_l2
        self.param_l1       = param_l1
        self.param_l2       = param_l2
        self.mode_do        = mode_do
        self.param_do       = param_do
        self.lr             = lr
        # Argumentos para entrenar el modelo
        self.dataset            = dataset
        self.patience           = patience
        self.epochs             = epochs
        self.batch_size         = batch_size
        # Argumentos para guardar el modelo
        self.pth_save_model     = pth_save_model
        self.pth_save_history   = pth_save_history

    def check_values(self) -> None:
        """
        Verifica los posibles errores y advertencias al ingresar los argumentos de la clase CreateFullAuto.

        Esta función no espera argumentos ni devuelve valores.
        """
        if self.verify == 'y':    
            method_menssage(self.cheack_values.__name__, 'Verifica los posibles errores y advertencias al ingresar los argumentos de la clase CreateFullAuto')
            # Evalua la viariable kernels
            kernel_label = 'número inicial de kernels'
            ve().check_type(self.kernels, int, kernel_label)        # Si es entero
            ve().check_positive(self.kernels, kernel_label)         # Si es mayor a cero
            ve().check_par(self.kernels, kernel_label)              # Si es par
            vw().check_limits(self.kernels, 8, 32, kernel_label)    # Advierte si el valor inicial no es recomendable

            # Evalua la variable dim
            dim_label = 'dimensión de los datos de entrada'
            ve().check_type(self.dim, int, dim_label)

            # Evalua la variable mode_l1, mode_l2, mode_do
            modes =  ['all', 'random', None]                                                # Lista que contiene los posibles valores para las variables mode_l1, mode_l2 y mode_do
            ve().check_arguments(self.mode_l1, modes, 'modo de uso de regularización l1')   # Evalua la variable mode_l1
            ve().check_arguments(self.mode_l2, modes, 'modo de uso de regularización l2')   # Evalua la variable mode_l2
            ve().check_arguments(self.mode_do, modes, 'modo de uso de drop out')           # Evalua la variable mode_do
        elif self.verify == 'n':
            pass
        else:
            ve().check_arguments(self.verify, ['y', 'n'], 'validación de argumentos')

    def addreg(self) -> regularizers:
        """
        Devuelve las instancias de regularización de keras l1, l2 y l1l1.

        Returns:
            regularizers: Objeto regularizador.
        """
        add_l1      = regularizers.L1(self.param_l1)                    # Define el regularizador l1 para la capa
        add_l2      = regularizers.L2(self.param_l2)                    # Define el regularizador l2 para la capa
        add_l1l2    = regularizers.L1L2(self.param_l2, self.param_l2)   # Define el regularizador l1 y l2 para la misma capa
    
        return add_l1, add_l2, add_l1l2
    
    def adddo(self) -> Union[Dropout, None]:
        """
        Devuelve las capa con drop out, dado el modo de uso y el valor del parámetro.

        Returns:
            Dropout:    Capa con drop out.
            None:       No agrega la capa.
        """
        if self.mode_do == 'all' or self.mode_do == 'random':       # Evalua la condición que debe cumplir el valor de la variable mode_do para añadir drop out
            return self.full_auto.add(Dropout(rate=self.param_do))  # Define la capa con drop out
        else:
            return None
        
    def rand_bol(self) -> random:
        """
        Devuelve True o False de manera aleatoria.

        Returns:
            random: Valor True o False.
        """
        return random.choice([True, False]) # Objeto random con posibilidad de devolver True o False
    
    def choice_reg(self) -> None:
        """
        Añade los objetos de regularización a las capas, apoyandose de la función add_reg y condicionales para aplicar
        la regularización pertinente o no hacerlo.

        Esta función no espera argumentos ni devuelve valores.
        """
        if self.mode_l1 == 'all' and self.mode_l2 == 'all':
            _, _, reg = self.addreg()
        elif self.mode_l1 == 'all' and self.mode_l2 == None:
            reg, *_ = self.addreg()
        elif self.mode_l1 == None and self.mode_l2 == 'all':
            _, reg, _ = self.addreg()
        elif self.mode_l1 == 'random' and self.mode_l2 == 'random':
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
        elif self.mode_l1 == 'random' and self.mode_l2 == None:
            l1 = self.rand_bol()
            if l1 == True:
                reg, *_ = self.addreg()
            else:
                reg = None
        elif self.mode_l1 == None and self.mode_l2 == 'random':
            l2 = self.rand_bol()
            if l2 == True:
                _, reg, _ = self.addreg()
            else:
                reg = None
        elif self.mode_l1 == 'random' and self.mode_l2 == 'all':
            l1 = self.rand_bol()
            if l1 == True:
                _, _, reg, _ = self.newlay()
            else:
                _, reg, _ = self.addreg()
        elif self.mode_l1 == 'all' and self.mode_l2 == 'random':
            l2 = self.rand_bol()
            if l2 == True:
                _, _, reg = self.addreg()
            else:
                reg, *_ = self.addreg()

    def choice_do(self):
        if self.mode_do == 'all':
            do = self.adddo()
        elif self.mode_do == 'random':
            choice = self.rand_bol()
            if choice == True:
                do = self.adddo()
            else:
                do = None
        elif self.mode_do == None:
            do = None

    def create_model(self) -> models.Sequential:
        """
        Construye la arquitectura de la red dados los parámetros de la clase CreateFullAuto, guardandolos en el modelo
        "full_auto_encoder"

        Returns:
            models.Sequential: Objeto que contiene el modelo secuencial de ren neuronal.
        """
        self.check_values()
        method_menssage(self.create_model.__name__, 'Define la arquitectura y la almacena en un modelo secuencial.')
        for lay in range(1, self.number_layers+1):
            if lay == 1 :
                self.full_auto.add(layers.Conv2D(self.kernels, (3, 3), activation='relu', padding='same', input_shape=(self.dim, self.dim, 3), kernel_regularizer=self.choice_reg()))
                self.choice_do()
                self.full_auto.add(layers.MaxPooling2D((2, 2)))
                self.kernels *= 2
            elif lay == self.number_layers:
                self.full_auto.add(layers.Conv2D(self.kernels, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg()))
                self.choice_do()
                self.full_auto.add(layers.MaxPooling2D((2, 2)))
                self.kernels -= self.kernels//2
            elif not lay in [1, self.number_layers]:
                self.full_auto.add(layers.Conv2D(self.kernels, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg()))
                self.choice_do()
                self.full_auto.add(layers.MaxPooling2D((2, 2)))
                self.kernels *= 2
        for lay in range(1, self.number_layers+1):
            if lay == self.number_layers:
                self.full_auto.add(layers.Conv2DTranspose(self.kernels, (2, 2), activation='relu', kernel_regularizer=self.choice_reg()))
                self.choice_do()
                self.full_auto.add(layers.UpSampling2D((2,2)))
                self.full_auto.add(layers.Conv2DTranspose(3, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg()))
            else:
                self.full_auto.add(layers.Conv2DTranspose(self.kernels, (3, 3), activation='relu', padding='same', kernel_regularizer=self.choice_reg()))
                self.choice_do()
                self.full_auto.add(layers.UpSampling2D((2,2)))
            self.kernels -= self.kernels//2

        optimizer_autoencoder = optimizers.Adam(learning_rate=self.lr)                      # Define el optimizador
        self.full_auto.compile(optimizer=optimizer_autoencoder, loss='mse', metrics='mse')  # Compila el modelo

        return self.full_auto
    
    def train_model(self) -> Union[models.Sequential, History]:
        """
        Define el mecanismo de parada temprana, entrena el modelo y guarda los datos del entrenamiento.

        Returns:
            models.sequential:  Modelo entrenado.
            History:            Información del entrenamiento.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        self.history = self.full_auto.fit(self.dataset, self.dataset, epochs=self.epochs, batch_size=self.batch_size, shuffle=False, validation_split=0.20, verbose=0, callbacks=[early_stopping])
        
        return self.full_auto, self.history
    
    def save_model(self) -> None:
        """
        Guarda el modelo que se ecuentre almacenado en la variable self.full_auto en formato h5.

        Esta función no espera argumentos ni devuelve valores.
        """
        pth_save = create_path_save(self.pth_save_model, 'full_auto_encoder', 'h5') # Define la ruta donde se guardará el archivo
        self.full_auto.save(pth_save)                                               # Guarda el modelo
        print(f'\nModelo guardado con éxito en "{pth_save}".\n')

    def save_history(self) -> None:
        """
        Guarda la información del entrenamiento como un archivo json.

        Esta función no espera argumentos ni devuelve valores.
        """
        history_dict = self.history.history
        pth_save = create_path_save(self.pth_save_history, 'train_history', 'json') # Define la ruta donde se guardará el archivo
        with open(pth_save, 'w') as file:                                           # Abre el archivo json
            json.dump(history_dict, file)                                           # Guarda el archivo