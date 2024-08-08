from resources.deploy import predict_som

path_sompy          = 'C:/camilo/uss/models/som/sompy_trained_tests.pkl'
path_dataset        = 'C:/camilo/uss/predicts/encoder/predicts_test.npy'
path_save_predicts  = 'C:/camilo/uss/predicts/classifier/'

predict_som(path_sompy, path_dataset, 3, path_save_predicts)