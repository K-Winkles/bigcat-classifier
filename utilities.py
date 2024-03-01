# Common imports
import os
import numpy as np
import tensorflow as tf
from typing import Tuple
from glob import glob

# Data Imports
import pandas as pd
from tqdm import tqdm
from tensorflow import image as tfi

# Data Visualization
import plotly.express as px
import matplotlib.pyplot as plt


def load_image(IMG_SIZE: tuple, image_path: str) -> tf.Tensor:

    '''
    The task of the function is to load the image present in the specified given image path. Loading the image the function also performed some 
    preprocessing steps such as resizing and normalization.
    
    Argument:
        image_path(str) : This is a string which represents the location of the image file to be loaded.
        
    Returns:
        image(tf.Tensor) : This is the image which is loaded from the given image part in the form of a tensor.
    '''

    # Check if image path exists
    assert os.path.exists(image_path), f'Invalid image path: {image_path}'

    # Load the image
    image = plt.imread(image_path)

    # Resize the Image
    image = tfi.resize(image, (IMG_SIZE, IMG_SIZE))

    # Convert image data type to tf.float32
    image = tf.cast(image, tf.float32)

    # Normalize the image to bring pixel values between 0 - 1
    image = image/255.0

    return image


def load_dataset(IMG_SIZE: tuple, root_path: str, class_names: list, batch_size: int = 32, buffer_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Load and preprocess images from the given root path and return them as numpy arrays.

    Args:
        root_path (str): Path to the root directory where all the subdirectories (class names) are present.
        class_names (list): List of the names of all the subdirectories (class names).
        batch_size (int): Batch size of the final dataset. Defaults to 32.
        buffer_size (int): Buffer size to use when shuffling the data. Defaults to 1000.

    Returns:
        Two numpy arrays, one containing the images and the other containing their respective labels.
    '''

    # Collect total number of data samples
    n_samples = sum([len(os.listdir(os.path.join(root_path, name))) for name in class_names])

    # Create arrays to store images and labels
    images = np.empty(shape=(n_samples, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    labels = np.empty(shape=(n_samples, 1), dtype=np.int32)

    # Loop over all the image file paths, load and store the images with respective labels
    n_image = 0
    for class_name in tqdm(class_names, desc="Loading"):
        class_path = os.path.join(root_path, class_name)
        for file_path in glob(os.path.join(class_path, "*")):
            # Load the image
            image = load_image(IMG_SIZE, file_path)

            # Assign label
            label = class_names.index(class_name)

            # Store the image and the respective label
            images[n_image] = image
            labels[n_image] = label

            # Increment the number of images processed
            n_image += 1

    # Shuffle the data
    indices = np.random.permutation(n_samples)
    images = images[indices]
    labels = labels[indices]

    return images, labels, class_names


def show_images(class_names, images: np.ndarray, labels: np.ndarray, n_rows: int=1, n_cols: int=5, figsize: tuple=(25, 8), model: tf.keras.Model=None) -> None:
    """
    Plots a grid of random images and their corresponding labels, with an optional prediction from a given model.

    Args:
        images (np.ndarray): Array of images to plot.
        labels (np.ndarray): Array of labels corresponding to the images.
        n_rows (int): Number of rows in the plot grid. Default is 1.
        n_cols (int): Number of columns in the plot grid. Default is 5.
        figsize (tuple): A tuple specifying the size of the figure. Default is (25, 8).
        model (tf.keras.Model): A Keras model object used to make predictions on the images. Default is None.

    Returns:
        None
    """

    # Loop over each row of the plot
    for row in range(n_rows):
        # Create a new figure for each row
        plt.figure(figsize=figsize)

        # Generate a random index for each column in the row
        rand_indices = np.random.choice(len(images), size=n_cols, replace=False)

        # Loop over each column of the plot
        for col, index in enumerate(rand_indices):
            # Get the image and label at the random index
            image = images[index]
            label = class_names[int(labels[index])]

            # If a model is provided, make a prediction on the image
            if model:
                prediction = model.predict(np.expand_dims(tf.squeeze(image), axis=0), verbose=0)[0]
                label += f"\nPrediction: {class_names[np.argmax(prediction)]}"

            # Plot the image and label
            plt.subplot(1, n_cols, col+1)
            plt.imshow(image)
            plt.title(label.title())
            plt.axis("off")

        # Show the row of images
        plt.show()


def plot_class_distribution(dir, class_names, set_name):
    class_dis = [len(os.listdir(dir + name)) for name in class_names]
    pie_chart = px.pie(values=class_dis, names=class_names, color=class_names, title=f"Class Distribution for {set_name}")
    pie_chart.show()