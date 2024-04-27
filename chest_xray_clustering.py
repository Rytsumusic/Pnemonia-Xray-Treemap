import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import rich.progress
from keras import layers, models, preprocessing, applications

model = models.load_model('./models/VGG16.hdf5')
model.summary()
print("sucessfully loaded model",model.output_shape)
embedding_output = model.get_layer('global_average_pooling2d_1').output
embedding_model = models.Model(inputs=model.input, outputs=embedding_output)
print("embedding shape",embedding_output.shape)

image_paths = list(glob.glob('..\\Pnemonia-Xray-Treemap\\chest_xray\\test\\**\\*.jpeg',recursive=True))
with rich.progress.Progress() as progress:
    task = progress.add_task("[red]Embedding images...", total=len(image_paths))
    for f in image_paths:
        print("number of images",len(image_paths))
        img = preprocessing.image.load_img(f, target_size=(224, 224))
        img_array = preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.vgg16.preprocess_input(img_array)
        embedding = embedding_model.predict(img_array)
        progress.update(task, advance=1)
        print(embedding)


