from resources.analize import AnalizeFullAuto as afa
from resources.create_architecture import CreateFullAuto as cfa
save = 'n'

# Carga el modelo y revisa la arquitectura
model = cfa().load_any_model('n', 'C:/camilo/uss/models/full_auto_encoder/full_auto_encoder_trained_loss0.0002978782285936177_22-08-2024_11-53_AM.keras')
print(model.summary())

# Analiza los datos del historial de entrenamiento
path_history            = 'C:/camilo/uss/histories/full_auto_encoder/train_history_22-08-2024_11-53_AM.json'
path_save_history_image = 'C:/camilo/uss/images/'
afa().plot_histories(path_history, save, path_save_history_image)

# Implementa el modelo con una imágen y compara los resultados
path_image              = 'C:/camilo/uss/data/data_tests/img.tif'
dim                     = 25
path_model              = 'C:/camilo/uss/models/full_auto_encoder/full_auto_encoder_trained_loss0.0002978782285936177_22-08-2024_11-53_AM.keras'
path_save_predict_image = 'C:/camilo/uss/images/'
afa().analize_predict_full_auto(path_image, 25, path_model, save, path_save_predict_image)

