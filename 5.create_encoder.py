from resources.create_architecture import CreateClassifier as cc
from resources.create_architecture import CreateFullAuto as cfa

path_full_auto      = 'C:/camilo/uss/models/full_auto_encoder/full_auto_encoder_trained_loss0.0002978782285936177_22-08-2024_11-53_AM.keras'
path_save_encoder   = 'C:/camilo/uss/models/encoder/'

encoder = cc().create_encoder(path_full_auto)                   # Crea el encoder
cfa().save_model('y', encoder, 'encoder', path_save_encoder)    # Guarda el encoder en un archivo keras
full_model = cfa().load_any_model('y', path_full_auto)
print(full_model.summary())
print(encoder.summary())

