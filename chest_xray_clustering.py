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
        embeddings.append(embedding)  # Append the embedding to the list
        progress.update(task, advance=1)

embeddings = np.concatenate(embeddings, axis=0)
        
print(embeddings)

# HDBSCAN
if len(embeddings) > 1:
    HDBSCAN = sklearn.cluster.HDBSCAN(min_cluster_size=5, metric='euclidean').fit(embeddings)
    clusters = HDBSCAN.fit_predict(embeddings)
else:
    print("Not enough samples for HDBSCAN clustering.")
print("clusters",clusters)

# TSNE
tsne = TSNE(n_components=2, perplexity=70).fit_transform(embeddings)
# plot actual images
fig, ax = plt.subplots()
for i, txt in enumerate(image_paths):
    img = preprocessing.image.load_img(txt, target_size=(224, 224))
    img = np.array(img)
    ax.imshow(img, extent=(tsne[i, 0], tsne[i, 0]+0.5, tsne[i, 1], tsne[i, 1]+0.5))
    #ax.annotate("", (tsne[i, 0], tsne[i, 1]))  # Remove the txt argument to remove the path annotation
ax.set_xlim(tsne[:, 0].min(), tsne[:, 0].max())
ax.set_ylim(tsne[:, 1].min(), tsne[:, 1].max())
# remove axis
ax.axis("off")
fig.savefig("tsne.png", dpi=1200)
#plt.scatter(tsne[:, 0], tsne[:, 1])
#plt.savefig("tsne.png", dpi=1200)
plt.show()

