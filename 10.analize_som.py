from resources.analize import AnalizeSOM as asm
import numpy as np

path_predicts       = 'C:/camilo/uss/predicts/classifier/som_predicts_2_c3.npy'
path_save_image     = 'C:/camilo/uss/images/'
som_predicts = np.load(path_predicts)
bmu = som_predicts[0]
classification = som_predicts[1]

asm().plot_matriz(bmu, (230,150), None, 'y', path_save_image, 'plot_bmu')
asm().plot_matriz(bmu, (230,150), ['grey', 'green', 'blue', 'black'], 'y', path_save_image, 'plot_classes')