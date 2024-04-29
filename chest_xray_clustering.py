import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import rich.progress
from keras import layers, models, preprocessing, applications
from sklearn.metrics import pairwise_distances
from pyclustering.cluster.kmedoids import kmedoids


import sklearn
from sklearn.cluster import HDBSCAN
from sklearn.manifold import TSNE
import squarify

model = models.load_model('./models/VGG16.hdf5')
model.summary()
print("sucessfully loaded model",model.output_shape)
embedding_output = model.get_layer('global_average_pooling2d_1').output
embedding_model = models.Model(inputs=model.input, outputs=embedding_output)
print("embedding shape",embedding_output.shape)


#image_paths = list(glob.glob('..\\Pnemonia-Xray-Treemap\\chest_xray\\test\\**\\*.jpeg',recursive=True))
image_paths = list(glob.glob('../Pnemonia-Xray-Treemap/chest_xray/test/**/*.jpeg',recursive=True))


embeddings = []  
with rich.progress.Progress() as progress:
    task = progress.add_task("[red]Embedding images...", total=len(image_paths))
    for f in image_paths:
        print("number of images",len(image_paths))
        img = preprocessing.image.load_img(f, target_size=(224, 224))
        img_array = preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.vgg16.preprocess_input(img_array)
        embedding = embedding_model.predict(img_array)
        embeddings.append(embedding)
        progress.update(task, advance=1)


        
print(embeddings)

# Clustering
if len(embeddings) > 1:
    # Reshape embeddings to have 2 dimensions
    embeddings_2d = np.squeeze(embeddings)
    embeddings_2d = embeddings_2d.reshape(embeddings_2d.shape[0], -1)
    k = 5  # Number of clusters
    kmedoids_instance = kmedoids(embeddings_2d, initial_index_medoids=np.random.choice(len(embeddings_2d), k, replace=False))
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()
else:
    print("Not enough samples for clustering.")

print("clusters", clusters)
print("medoids", medoids)


tsne = TSNE(n_components=2, perplexity=70).fit_transform(embeddings_2d)


fig, ax = plt.subplots()
for i, txt in enumerate(image_paths):
    img = preprocessing.image.load_img(txt, target_size=(224, 224))
    img = np.array(img)
    ax.imshow(img, extent=(tsne[i, 0], tsne[i, 0]+0.5, tsne[i, 1], tsne[i, 1]+0.5))
    #ax.scatter(tsne[i, 0], tsne[i, 1], marker='o', color='blue')
    for medoid in medoids:
        ax.scatter(tsne[medoid, 0], tsne[medoid, 1], marker='o', color='red')

ax.set_xlim(tsne[:, 0].min() - 1, tsne[:, 0].max() + 1)
ax.set_ylim(tsne[:, 1].min() - 1, tsne[:, 1].max() + 1)
ax.axis("off")
ax.set_title("Clustering with medoids labeled")
fig.savefig("Clustering with medoids labeled", dpi=1200)
plt.show()







# Treemap
fig, ax = plt.subplots(figsize=(10, 10))
sizes = [1] * len(medoids)  
colors = ['blue', 'green', 'red', 'yellow', 'orange'][:len(medoids)]  # Add more colors if needed
labels = [''] * len(medoids)  
for i, medoid in enumerate(medoids):
    img = preprocessing.image.load_img(image_paths[medoid], target_size=(224, 224))
    img = np.array(img)
    ax.imshow(img, extent=(0, 1, 0, 1))
    labels[i] = 'Medoid {}'.format(i)
squarify.plot(sizes=sizes, label=labels, color=colors, alpha=0.8)

plt.axis('off')

plt.savefig('treemap.png', dpi=1200)
plt.show()