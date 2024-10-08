from sompy.sompy import SOMFactory
from resources.create_architecture import CreateClassifier as cc
import numpy as np

sompy = SOMFactory().build(data             = np.load('C:/camilo/uss/predicts/encoder/predicts_encoder_1.npy'), 
                           mapsize          = [2,2], 
                           mask             = None, 
                           mapshape         = 'planar', 
                           lattice          = 'rect', 
                           normalization    = 'var', 
                           initialization   = 'pca', 
                           neighborhood     = 'gaussian', 
                           training         = 'batch', 
                           name             = 'clasificador', 
                           component_names  = ['agua', 'vegetación', 'edificaciones'])

path_save_sompy = 'C:/camilo/uss/models/som/'
cc().save_sompy(sompy, 'sompy', path_save_sompy)
