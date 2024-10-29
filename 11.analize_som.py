from resources.analize import AnalizeSOM as asm
import numpy as np

path_predicts       = 'C:/camilo/uss/predicts/classifier/som_22-08-2024_01-46_PM.npy'
path_save_image     = 'C:/camilo/uss/images/'
som_predicts = np.load(path_predicts)
bmu = som_predicts[0]
classification = som_predicts[1]

asm().plot_matriz(bmu, (460,300), None, 'n', path_save_image, 'plot_bmu')
asm().plot_matriz(bmu, (460,300), ['green', 'blue', 'gray', 'black'], 'n', path_save_image, 'plot_classes')