import tensorflow as tf
import tensorflow_io as tfio # Used for decoding DICOM files
from tensorflow.keras.models import load_model # Used for loading the models
from tensorflow.keras.models import Model # Used to cut the Model
import matplotlib.pyplot as plt # Used to visualize the sample
import numpy as np # For pre-processing the sample




model = load_model('./models/Xception.hdf5')


embedding_output = model.get_layer('global_average_pooling2d').output
embedding_model = Model(inputs=model.input, outputs=embedding_output)
