import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from zipfile import ZipFile


#TODO
#   experiment with different number of layers and activation functions etc
#   create nice graphics for outputs, and look at different metrics
#   results are very poor- probably overfitting, also probably because there are so many target categories
#   maybe also put color filter in preporocessing step
#   make dropout and preprocessing into params and run seeveral models using different techniques then we know what gives the best resutls

def model(num_classes,img_height,img_width):

    model = Sequential([

        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        # Data Augmentation
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),

        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.AveragePooling2D(),
        layers.Dropout(0.05),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.AveragePooling2D(),
        layers.Dropout(0.1),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.AveragePooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    return model

def visualize_results(epochs):


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    with open("results_sept12", 'w') as f:
        print('test accuracies: {}'.format(acc), file=f)
        print('valid accuracies: {}'.format(val_acc), file=f)

        print('test loss: {}'.format(loss), file=f)
        print('valid loss: {}'.format(val_loss), file=f)




def fit(epochs):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs
    )
    return history

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #"params" so to speak- how big should my images actually be...
    height = 256
    width = 256

    #change this when i want to run it on full datasett
    testing = False
    #change this, how many iterations do we want
    epochs = 10

    if testing == True:

        train_data = tf.keras.preprocessing.image_dataset_from_directory("birds_SmallSet/train/", image_size=(height, width))
        val_data = tf.keras.preprocessing.image_dataset_from_directory("birds_SmallSet/valid/", image_size=(height, width))
    else:

        zf = ZipFile('birds.zip', 'r')
        zf.extractall()
        zf.close()

        train_data = tf.keras.preprocessing.image_dataset_from_directory("birds/train/", image_size=(height, width))
        val_data = tf.keras.preprocessing.image_dataset_from_directory("birds/valid/", image_size=(height, width))




    num_classes = len(train_data.class_names)

    model = model(num_classes,height,width)

    history = fit(epochs)

    visualize_results(epochs)




