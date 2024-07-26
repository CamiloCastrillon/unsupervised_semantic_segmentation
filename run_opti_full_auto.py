from resources.optimize import OptimizeFullAuto as ofa
import numpy as np

path = 'C:/camilo/resources_uss/datasets/autoencoder/dataset50.npy'
dataset = np.load(path)
params = {
    'dataset'   : dataset,
    'nl_min'    : 2,
    'nl_max'    : 3,
    'dim'       : 50,
    'lr_min'    : 0.0001,
    'lr_max'    : 0.01,
    'ep_min'    : 50,
    'ep_max'    : 300,
    'ba_min'    : 16,
    'ba_max'    : 512,
    'ink'       : 8,
    'rl1_min'   : 0.0001,
    'rl1_max'   : 0.1,
    'rl2_min'   : 0.0001,
    'rl2_max'   : 0.1,
    'do_min'    : 0.1,
    'do_max'    : 0.3,
    'esp_min'   : 10,
    'esp_max'   : 20
}

ofa = ofa(params['dataset'], params['nl_min'], params['nl_max'], params['dim'], params['lr_min'], params['lr_max'], 
          params['ep_min'], params['ep_max'], params['ba_min'], params['ba_max'], params['ink'], params['rl1_min'], params['rl1_max'],
          params['rl2_min'], params['rl2_max'], params['do_min'], params['do_max'], params['esp_min'], params['esp_max'])

print(ofa.check_values())

