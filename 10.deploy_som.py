from resources.deploy import predict_som

path_sompy          = 'C:/camilo/uss/models/som/sompy_trained_[3, 3]_te0.0_22-08-2024_01-57_PM.pkl'
path_dataset        = 'C:/camilo/uss/datasets/som/predicts_22-08-2024_12-36_PM.npy'
path_save_predicts  = 'C:/camilo/uss/predicts/classifier/'
n_clases            = 3
predict_som(path_sompy, path_dataset, n_clases, path_save_predicts)