from resources.analize import AnalizeFullAuto as afa

save = 'y'

# Analiza los datos del historial de entrenamiento
path_history            = 'C:/camilo/uss/histories/full_auto_encoder/train_history_tests_0.json'
path_save_history_image = 'C:/camilo/uss/images/'
afa().plot_histories(path_history, save, path_save_history_image)

# Implementa el modelo con una imágen y compara los resultados
path_image              = 'C:/camilo/uss/data/img.tif'
dim                     = 50
path_model              = 'C:/camilo/uss/models/full_auto_encoder/full_auto_encoder_trained_tests_0.keras'
path_save_predict_image = 'C:/camilo/uss/images/'
afa().analize_predict_full_auto(path_image, 50, path_model, save, path_save_predict_image)