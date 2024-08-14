from resources.create_dataset import GenDataAutoencoder as gda
dim         = 51
pth_data    = 'C:/camilo/uss/data/data_aguacate/'
#'C:/camilo/uss/datasets/full_auto_encoder/data_tests/'
#'C:/camilo/uss/datasets/full_auto_encoder/data_aguacate/'
pth_save    = 'C:/camilo/uss/datasets/full_auto_encoder/'
#C:/camilo/uss/datasets/full_auto_encoder/
gda = gda(dim, pth_data, pth_save)
gda.gen_data()