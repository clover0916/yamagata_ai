"""
This file is used to train the model for clock reading.

The model is a CNN with 3 convolutional layers and 2 fully connected layers.

The data is stored in ./clocks/ directory.
clocks_all.txt contains the path to each image and the corresponding time of the clock in the image.
The format of the file is as follows:
    path/to/image1.jpg HH    MM   SS
    path/to/image2.jpg HH    MM   SS
HH, MM, SS represent the hour, minute, and second respectively.

This file reads the clocks_all.txt file, loads the images, and feeds them into the CNN.

The output of the CNN represents the predicted hour, and minute values.

The difference between the predicted hour, and minute values from the CNN and the actual hour, and minute values is used as the loss function.
"""

import cv2
from cycler import V
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import keras_tuner as kt
import matplotlib.pyplot as plt

IMAGE_SIZE = 116

# Load the data from clocks_all.txt
data_file = "./clocks_all.txt"
with open(data_file, "r") as file:
    data = file.readlines()

# Preprocess the data
images = []
labels = []
for line in data:
    line = line.strip().split()
    image_path = line[0]
    hour = int(line[1])
    minute = int(line[2])

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)

    images.append(image)
    labels.append((hour, minute))

# Convert the data to numpy arrays
images = np.array(images)
labels = np.array(labels)


# Define the CNN model
def build_model(hp):
    model = Sequential()
    model.add(
        Conv2D(
            hp.Int("filters_1", min_value=32, max_value=64, step=32),
            (3, 3),
            activation="relu",
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            hp.Int("filters_2", min_value=64, max_value=128, step=64),
            (3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            hp.Int("filters_3", min_value=128, max_value=256, step=128),
            (3, 3),
            activation="relu",
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(2, activation="linear"))

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["accuracy"],
    )

    return model


# Define the tuner
tuner = kt.Hyperband(
    build_model,
    objective="val_loss",
    max_epochs=5,
    directory="keras_tuner",
    project_name="clock_tuner",
)

# Perform the hyperparameter search
tuner.search(images, labels, epochs=5, validation_split=0.2)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

best_model.summary()

# Train the best model
history = best_model.fit(
    images,
    labels,
    epochs=100,
    validation_split=0.2,
    batch_size=32,
    verbose=1,
)

# Save the best model
best_model.save("trained_model.keras")

# Plot the metrics
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(history.history["loss"], label="loss", color="blue")
ax1.plot(history.history["val_loss"], label="val_loss", color="orange")
ax2.plot(history.history["accuracy"], label="accuracy", color="green")
ax2.plot(
    history.history["val_accuracy"],
    label="val_accuracy",
    color="red",
)
ax2.set_ylim([0, 1])

handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax1.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.0)

plt.title("Model Metrics")
plt.xlabel("Epoch")
plt.savefig("metrics_curve.png")
plt.show()
