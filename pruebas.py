from create_dataset_autoencoder import GenDataAutoencoder as gda

stack = []
for data in gda.get_imgs():
    img, w, h = data
    sections = gda.make_data(img, w, h, 50)
    stack.append(sections)

gda.define_data('dataset1', stack)

