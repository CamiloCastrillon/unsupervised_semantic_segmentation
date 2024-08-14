from resources.optimize import OptimizeSOM as osm

path_dataset            = 'C:/camilo/uss/predicts/encoder/predicts_encoder_1.npy'
mapsize                 = [2,2]
mask                    = None
mapshape                = ['planar']
lattice                 = ['rect']
normalization           = ['var']
initialization          = ['pca', 'random']
neighborhood            = ['gaussian', 'bubble']
training                =  ['seq', 'batch']
name                    = 'clasificador'
component_names         = ['agua', 'vegetación', 'edificaciones']
train_rough_len_min    = 10
train_rough_len_max    = 30
train_finetune_len_min = 5
train_finetune_len_max = 15
n_trials               = 3
umbral                 = 0.001
path_save_params       = 'C:/camilo/uss/best_params/som/'
path_save_models       = 'C:/camilo/uss/models/som/'

osm = osm(  path_dataset, mapsize, mask, mapshape, lattice, normalization, initialization, neighborhood, training, name, component_names, 
            train_rough_len_min, train_rough_len_max, train_finetune_len_min, train_finetune_len_max, n_trials, umbral, path_save_params, path_save_models)

osm.execute_study()