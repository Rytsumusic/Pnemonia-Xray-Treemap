import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import rich.progress
from keras import layers, models, preprocessing, applications
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


image_paths = list(glob.glob('..\\Pnemonia-Xray-Treemap\\chest_xray\\test\\**\\*.jpeg',recursive=True))
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

embeddings = np.concatenate(embeddings, axis=0)
        
print(embeddings)

# Clustering
if len(embeddings) > 1:
    HDBSCAN = sklearn.cluster.HDBSCAN(min_cluster_size=2, metric='euclidean').fit(embeddings)
    clusters = HDBSCAN.fit_predict(embeddings)
    
else:
    print("Not enough samples for HDBSCAN clustering.")
print("clusters",clusters)





tsne = TSNE(n_components=2, perplexity=70).fit_transform(embeddings)

# Regular clustering mapping to show the clustering works. Centroids are marked with red X.
fig, ax = plt.subplots()
for i, txt in enumerate(image_paths):
    img = preprocessing.image.load_img(txt, target_size=(224, 224))
    img = np.array(img)
    ax.imshow(img, extent=(tsne[i, 0], tsne[i, 0]+0.5, tsne[i, 1], tsne[i, 1]+0.5))

centroids = np.zeros((len(np.unique(clusters)), 2))
for i, cluster_label in enumerate(np.unique(clusters)):
    cluster_points = tsne[clusters == cluster_label]
    centroid = np.mean(cluster_points, axis=0)
    centroids[i] = centroid
    ax.scatter(centroid[0], centroid[1], marker='o', color='Green',)
    
ax.set_xlim(tsne[:, 0].min(), tsne[:, 0].max())
ax.set_ylim(tsne[:, 1].min(), tsne[:, 1].max())
ax.axis("off")
fig.savefig("tsne.png", dpi=1200)
plt.show()



clusters = [c for c in clusters if c >= 0]
cluster_sizes = np.bincount(clusters)
cluster_labels = np.unique(clusters)

normalized_sizes = cluster_sizes / cluster_sizes.sum()

fig, ax = plt.subplots(figsize=(10, 10))
squarify.plot(sizes=normalized_sizes, label=cluster_labels, alpha=0.8, ax=ax)

#for i, centroid in enumerate(centroids):
 #   img = preprocessing.image.load_img(image_paths[i], target_size=(224, 224))
  #  img = np.array(img)
   # ax.imshow(img, extent=(centroid[0]-0.25, centroid[0]+0.25, centroid[1]-0.25, centroid[1]+0.25))
    #ax.scatter(centroid[0], centroid[1])

plt.axis('off')
plt.title('Cluster Treemap')
plt.savefig('treemap.png', dpi=1200)
plt.show()