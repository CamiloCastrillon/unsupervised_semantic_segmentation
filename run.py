from create_dataset_autoencoder import GenDataAutoencoder as gda

dim         = 2000
pth_data    = 'C:/camilo/resources_uss/data/'
pth_save    = 'C:/camilo/resources_uss/datasets/autoencoder/' 
gda = gda(dim, pth_data, pth_save)

gda.gen_data()