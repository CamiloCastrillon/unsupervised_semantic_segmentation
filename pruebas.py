from resources.create_architecture import CreateClassifier as cc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

path_sompy      = 'C:/camilo/uss/models/som/sompy_trained_tests.pkl'
path_dataset    = 'C:/camilo/uss/predicts/encoder/predicts_test.npy'
dataset = np.load(path_dataset)
sompy   = cc().load_sompy(path_sompy)

bmu             = sompy.project_data(dataset)
neuron_weights  = sompy.codebook.matrix

bmu_weights = []
for winning_neuron in bmu:
    weight = neuron_weights[winning_neuron]
    bmu_weights.append(weight)
    
bmu_weights = np.array(bmu_weights)

kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(bmu_weights)

labels = kmeans.labels_

labels_reshaped = labels.reshape((230, 150))
cmap = plt.get_cmap('tab20')  # Cambia 'tab20' según tus preferencias

# Crear la figura y el eje
plt.figure(figsize=(11.5, 7.5))
plt.imshow(labels_reshaped, cmap=cmap, interpolation='nearest')

# Agregar una barra de colores para mostrar el mapeo de valores a colores
plt.colorbar(label='Valor')

# Mostrar la imagen
plt.title('Imagen con colores para valores diferentes')
plt.xlabel('Ancho')
plt.ylabel('Alto')
plt.show()
