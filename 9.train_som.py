from resources.create_architecture import CreateClassifier as cc
import os 

# Carga el modelo
path_sompy = 'C:/camilo/uss/models/som/sompy_2.pkl'
sompy = cc().load_sompy(path_sompy)

# Entrena el modelo
_train_rough_len    = 23
_train_finetune_len = 11
sompy.train(n_job=os.cpu_count()-4, verbose=False, train_rough_len = _train_rough_len, train_finetune_len = _train_finetune_len)

# Guarda el modelo entrenado
path_save_sompy = 'C:/camilo/uss/models/som/'
cc().save_sompy(sompy, 'sompy_trained', path_save_sompy)