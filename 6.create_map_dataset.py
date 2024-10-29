from resources.deploy import predict_sequential

path_encoder        = 'C:/camilo/uss/models/encoder/encoder_22-08-2024_12-34_PM.keras'
path_dataset        = 'C:/camilo/uss/datasets/full_auto_encoder/dataset_dim25_tests.npy'
path_save_predicts  = 'C:/camilo/uss/datasets/som/'

predict_sequential(path_encoder, path_dataset, path_save_predicts)