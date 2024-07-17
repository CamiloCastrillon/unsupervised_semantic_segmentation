from keras._tf_keras.keras import layers, optimizers, models, regularizers
from keras._tf_keras.keras.layers import Dropout
from resources.verify_variables import VerifyErrors as ve, VerifyWarnings as vw
from resources.message import method_menssage
import random
from typing import Union

class CreateFullAuto:
    """
    Crea y guarda la arquitectura la arquitectura compilada y sin entrenar del full auto encoder, en un archivo h5,
    según la estrcutura de este trabajo.

    Args:
        kernels (int):              Número de kernels con el que se crea la capa inicial.
        dim (int):                  Dimensión m de los datos de entrada (m,m,3)
        number_layers (int):        Número de capas del encader.
        mode_l1 (Union[str,None]):  Modo de uso de regularización l1.
            - 'all':    Todas las capas tendrán regularización l1.
            - 'random': Capas elegidas aleatoriamente tendrán regularización l1.
            - None:     Ninguna capa tendrá regularización l1.
        mode_l2 (Union[str,None]):  Modo de uso de regularización l2.
            - 'all':    Todas las capas tendrán regularización l2.
            - 'random': Capas elegidas aleatoriamente tendrán regularización l2.
            - None:     Ninguna capa tendrá regularización l1.
        param_l1 (float):           Valor de regularización l1.
        param_l2 (float):           Valor de regularización l2.
        mode_do (Union[str,None]):  Modo de uso de drope out.
            - 'all':    Todas las capas tendrán drope out.
            - 'random': Capas elegidas aleatoriamente tendrán drope out.
            - None:     Ninguna capa tendrá drope out.
        lr (float):                 Valor de learning rate.
    
    Returns:
        None: No se espera argumento de salida.
    """
    def __init__(self, kernels:int, dim:int, number_layers:int, mode_l1:Union[str,None], mode_l2:Union[str,None], param_l1:float, param_l2:float, mode_do:Union[str,None], param_do:float, lr:float) -> None:
        
        self.full_auto      = models.Sequential(name='full_autoencoder')
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

    def check_values(self) -> None:
        """
        Verifica los posibles errores y advertencias al ingresar los argumentos de la clase CreateFullAuto.

        Esta función no espera argumentos ni devuelve valores.
        """
        method_menssage(self.cheack_values.__name__, 'Verifica los posibles errores y advertencias al ingresar los argumentos de la clase CreateFullAuto')
        modes =  ['all', 'random', None]                                                # Lista que contiene los posibles valores para las variables mode_l1, mode_l2 y mode_do
        ve().check_arguments(self.mode_l1, modes, 'modo de uso de regularización l1')   # Evalua la variable mode_l1
        ve().check_arguments(self.mode_l2, modes, 'modo de uso de regularización l2')   # Evalua la variable mode_l2
        ve().check_arguments(self.mode_do, modes, 'modo de uso de drope out')           # Evalua la variable mode_do

    def addreg(self) -> regularizers:
        """
        Devuelve las instancias de regularización de keras l1, l2 y l1l1.

        Returns:
            regularizers: Objeto regularizador.
        """
        add_l1      = regularizers.L1(self.param_l1)
        add_l2      = regularizers.L2(self.param_l2)
        add_l1l2    = regularizers.L1L2(self.param_l2, self.param_l2)
    
        return add_l1, add_l2, add_l1l2
    
    def adddo(self):
        if self.mode_do == 'all' or self.mode_do == 'random':
            return self.full_auto.add(Dropout(rate=self.param_do))
        else:
            return None
        
    def rand_bol(self):
        return random.choice([True, False])
    
    def choice_reg(self):
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

    def create_model(self):
        self.check_values()
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
        return self.full_auto

    def compile_model(self):
        optimizer_autoencoder = optimizers.Adam(learning_rate=self.lr)
        self.full_auto.compile(optimizer=optimizer_autoencoder, loss='mse', metrics='mse')
