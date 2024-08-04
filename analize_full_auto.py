from resources.analize import AnalizeFullAuto as afa

# Analiza los datos del historial de entrenamiento
path_history            = 'C:/camilo/uss/histories/full_auto_encoder/history_prueba.json'
path_save_history_image = 'C:/camilo/uss/images/'
afa().plot_histories(path_history, path_save_history_image)

# Implementa el modelo con una imágen y compara los resultados
path_image              = 'C:/camilo/uss/data/img.tif'
dim                     = 50
path_model              = 'C:/camilo/uss/models/full_auto_encoder/full_auto_encoder_pruebas.keras'
path_save_predicts      = 'C:/camilo/uss/predicts/full_auto_encoder/'
path_save_predict_image = 'C:/camilo/uss/images/'
afa().one_predict(path_image, 50, path_model, path_save_predicts, path_save_predict_image)