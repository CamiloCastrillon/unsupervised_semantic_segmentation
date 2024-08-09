from resources.deploy import predict_sequential

path_encoder        = 'C:/camilo/uss/models/encoder/encoder_1.keras'
path_dataset        = 'C:/camilo/uss/datasets/full_auto_encoder/dataset_dim50.npy'
path_save_predicts  = 'C:/camilo/uss/predicts/encoder/'

predict_sequential(path_encoder, path_dataset, path_save_predicts)