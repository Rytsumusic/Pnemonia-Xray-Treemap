import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
model = keras.load_model('./models/Xception.hdf5')
embedding_output = model.get_layer('global_average_pooling2d').output
embedding_model = keras.Model(inputs=model.input, outputs=embedding_output)

# Load and preprocess the images
image_files = glob.glob('path_to_your_images/*/*')  
images = []
for f in image_files:
    image = keras.model.load_img(f, target_size=(299, 299))  # adjust target_size to match your model's input size
    image = keras.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    images.append(image)

# Create the embeddings
embeddings = embedding_model.predict(np.vstack(images))


