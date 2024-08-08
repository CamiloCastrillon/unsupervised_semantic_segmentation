from resources.optimize import OptimizeFullAuto as ofa
import numpy as np

# Define las variables
dataset        = np.load('C:/camilo/uss/datasets/full_auto_encoder/dataset_dim50.npy')
nl_min         = 2
nl_max         = 4
dim            = 50
lr_min         = 0.0001
lr_max         = 0.001
ep_min         = 50
ep_max         = 300 
ba_min         = 16
ba_max         = 512
ink            = 16
mode_l1        = None
rl1_min        = None
rl1_max        = None
mode_l2        = None
rl2_min        = None
rl2_max        = None
mode_do        = None
do_min         = None
do_max         = None
esp_min        = 10
esp_max        = 20
n_trials       = 5
save_param     = 'y'
pth_save_params= 'C:/camilo/uss/best_params/full_auto_encoder/'
umbral         = 0.001
pth_save_model = 'C:/camilo/uss/models/full_auto_encoder/'
pth_save_hist  = 'C:/camilo/uss/histories/full_auto_encoder/'

# Crea la instancia de la clase con los parámetros
ofa = ofa(dataset, nl_min, nl_max, dim, lr_min, lr_max, ep_min, ep_max, ba_min, ba_max, ink, mode_l1, rl1_min, rl1_max, mode_l2, rl2_min, rl2_max, 
          mode_do, do_min, do_max, esp_min, esp_max, n_trials, save_param, pth_save_params, umbral, pth_save_model, pth_save_hist)

# Crea el objeto de estudio optuna
ofa.execute_study()