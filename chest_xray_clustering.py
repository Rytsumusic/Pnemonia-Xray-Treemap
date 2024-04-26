import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # Used for loading the models
from tensorflow.keras.models import Model # Used to cut the Model
import matplotlib.pyplot as plt
 
 


model = load_model('./models/Xception.hdf5')
embedding_output = model.get_layer('global_average_pooling2d').output
embedding_model = Model(inputs=model.input, outputs=embedding_output)

# Load and preprocess the images
image_files = glob.glob('path_to_your_images/*/*')  
images = []
for f in image_files:
    image = load_img(f, target_size=(299, 299))  # adjust target_size to match your model's input size
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    images.append(image)

# Create the embeddings
embeddings = embedding_model.predict(np.vstack(images))


