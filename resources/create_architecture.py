import tensorflow as tf
from keras._tf_keras.keras import layers, optimizers, initializers, models, regularizers
from keras._tf_keras.keras.layers import Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
from resources.message import error_message, warning_message, method_menssage
import random

class CreateFullAuto:
    def __init__(self, kernels, dim, number_layers, mode_l1, mode_l2, param_l1, param_l2, mode_do, param_do, lr):
        
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

    def check_values(self):    
        # Evalua la variable mode_l1, mode_l2 y mode_do
        modes =  ['all', 'random', None]
        if not self.mode_l1 in modes:
            menssage = error_message('La variable modo de uso de regularización l1 debe tener los valores "all", "individual", "random", o None')
        elif not self.mode_l2 in modes:
            menssage = error_message('La variable modo de uso de regularización l2 debe tener los valores "all", "individual", "random", o None')
        elif not self.mode_do in modes:
            menssage = error_message('La variable modo de uso de regularización l2 debe tener los valores "all", "individual", "random", o None')
        else:
            menssage = ''
        
        return menssage

    def addreg(self):
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
