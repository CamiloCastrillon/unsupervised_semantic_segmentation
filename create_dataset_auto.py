from resources.create_dataset import GenDataAutoencoder as gda
dim         = 50
pth_data    = 'C:/camilo/uss/data/'
#C:/camilo/uss/data/
pth_save    = 'C:/camilo/uss/datasets/full_auto_encoder/'
#C:/camilo/uss/datasets/full_auto_encoder/
gda = gda(dim, pth_data, pth_save)
gda.gen_data()