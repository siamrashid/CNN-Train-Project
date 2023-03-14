import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential


# Downloading dataset from Google APIs
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file("flower_photos", origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

# Loading the Fashion MNIST dataset
(train_images, train_labels), (eval_images, eval_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images_reshaped = train_images.reshape(60000, 28, 28, 1)


def print_task_information():
    print("This task involves training multiple models and saving their details into a CSV file. The number of models "
          "to train and the hyperparameters of each model can be specified by the user.")


def build_model(num_layers, model_number, input_shape=(28, 28, 1)):
    # Initializing variables
    filters = []
    kernel_sizes = []
    activations = []
    pooling_sizes = []
    paddings = []

    # Prompting user for hyperparameters
    epochs = int(input(f"Enter the number of epochs for Model {model_number}: "))
    for i in range(num_layers):
        filters.append(int(input(f"Enter filter size for layer {i + 1}: ")))
        kernel_sizes.append(int(input(f"Enter kernel size for layer {i + 1}: ")))
        activations.append(input(f"Enter activation function for layer {i + 1}: "))
        pooling_sizes.append(int(input(f"Enter pooling size for layer {i + 1}: ")))
        paddings.append(input(f"Enter padding value for layer {i + 1}: "))

    # Building the model
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    for filter, kernel_size, activation, pool_size, padding in zip(filters, kernel_sizes, activations, pooling_sizes,
                                                                    paddings):
        model.add(Conv2D(filters=filter, kernel_size=(kernel_size, kernel_size), activation=activation,
                         padding=padding))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    history = model.fit(train_images_reshaped, train_labels, epochs=epochs, validation_split=0.4)

    return history, model, epochs


def main():
    print_task_information()
    num_models = int(input("Enter the number of models to train: "))

    # Creating a Pandas DataFrame to store the details of all models
    df = pd.DataFrame(columns=['Model Name', 'Architecture', 'Training Accuracy', 'Validation Accuracy',
                               'Training Loss', 'Validation Loss', 'Epochs'])

    for i in range(1, num_models + 1):
        print(f"\nDetails of architecture and hyperparameters for Model {i}:")
        num_layers = int(input("Enter the number of layers: "))
        history, model, epochs = build_model(num_layers, i)

        # Adding details of the trained model to the DataFrame
        model_summary = model.summary()
        model_name = model.name
        architecture = model_summary
        training_accuracy = max(history.history['accuracy'])
        val_accuracy = max
