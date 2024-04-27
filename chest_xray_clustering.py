import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import rich.progress
from keras import layers, models, preprocessing, applications

model = applications.VGG16(weights='imagenet', include_top=False)
image_paths = glob.glob('/chest_xray/test/**/*jpeg',recursive=True)

embeddings = []

with rich.progress.Progress() as progress:
    task = progress.add_task("[red]Embedding images...", total=len(image_paths))
    try:

        for image_path in image_paths:
            try:
                print(f"Processing image {image_path}")
                image = preprocessing.image.load_img(image_path, target_size=(224, 224))
                image = preprocessing.image.img_to_array(image)
                image = np.expand_dims(image, axis=0)
                image = applications.vgg16.preprocess_input(image)
                embedding = model.predict(image)
                embeddings.append(embedding)
                progress.update(task, advance=1)
                progress.refresh()
                print(embedding.shape)
            except Exception as e:
                print(f"Failed to process image {image_path}. Error: {e}")
    except Exception as e:
        print(f"Error occurred: {e}")

