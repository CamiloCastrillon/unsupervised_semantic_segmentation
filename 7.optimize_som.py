from resources.optimize import OptimizeSOM as osm
import os

hilos_pc = os.cpu_count()

n_hilos                 = 16
path_dataset            = 'C:/camilo/uss/datasets/som/predicts_22-08-2024_12-36_PM.npy'
mapsize                 = [3,3]
mask                    = None
mapshape                = ['planar']
lattice                 = ['rect']
normalization           = ['var']
initialization          = ['pca', 'random']
neighborhood            = ['gaussian', 'bubble']
training                = ['seq', 'batch']
name                    = 'clasificador'
component_names         = ['agua', 'vegetación', 'edificaciones']
train_rough_len_min    = 20
train_rough_len_max    = 50
train_finetune_len_min = 10
train_finetune_len_max = 30
n_trials               = 10
umbral                 = 0.001
path_save_params       = 'C:/camilo/uss/best_params/som/'
path_save_models       = 'C:/camilo/uss/models/som/'

osm = osm(  n_hilos, path_dataset, mapsize, mask, mapshape, lattice, normalization, initialization, neighborhood, training, name, component_names, 
            train_rough_len_min, train_rough_len_max, train_finetune_len_min, train_finetune_len_max, n_trials, umbral, path_save_params, path_save_models)

osm.execute_study()