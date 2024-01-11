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

The output of the CNN represents the predicted hour, minute, and second values.

The difference between the predicted hour, minute, and second values from the CNN and the actual hour, minute, and second values is used as the loss function.
"""

from cProfile import label
import cv2
from cycler import V
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
    second = int(line[3])

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)

    images.append(image)
    labels.append((hour, minute, second))

# Convert the data to numpy arrays
images = np.array(images)
labels = np.array(labels)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2,
)

# Generate the training data
train_generator = datagen.flow(
    images, labels, batch_size=32, subset="training", shuffle=True
)

# Generate the validation data
validation_generator = datagen.flow(
    images, labels, batch_size=32, subset="validation", shuffle=True
)


# Define the CNN model
model = Sequential()
model.add(
    Conv2D(64, (3, 3), activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
)
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dense(3, activation="linear"))

model.summary()

# Compile the model
model.compile(
    optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["accuracy"]
)

# Train the model
history = model.fit(train_generator, epochs=100, validation_data=validation_generator)

model.save("trained_model.keras")

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
