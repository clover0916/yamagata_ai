import random
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 116

# Load the model
model = load_model("trained_model.keras")

# Read the list of image files
data_file = "./clocks_all.txt"
with open(data_file, "r") as file:
    data = file.readlines()

# Select 10 random images
data = random.sample(data, 10)

# Initialize the plot
fig, axs = plt.subplots(5, 2, figsize=(10, 15))
fig.suptitle("Actual vs Predicted Times", fontsize=16)

# Perform prediction and calculate the error for the selected images
for i, line in enumerate(data):
    line = line.strip().split()
    image_path = line[0]
    hour = int(line[1])
    minute = int(line[2])
    second = int(line[3])

    # Load the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    # Perform prediction
    predictions = model.predict(img, verbose=0)

    predictions = np.round(predictions).astype(int)

    # Display the image in the subplot
    axs[i // 2, i % 2].imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    axs[i // 2, i % 2].axis("off")

    # Create the text to display
    actual_time = f"Actual: {hour:02d}:{minute:02d}:{second:02d}"
    predicted_time = f"Predicted: {predictions[0][0]:02d}:{predictions[0][1]:02d}:{predictions[0][2]:02d}"

    # Display the text in the subplot
    axs[i // 2, i % 2].text(
        0.5,
        -0.1,
        actual_time + "\n" + predicted_time,
        size=10,
        ha="center",
        transform=axs[i // 2, i % 2].transAxes,
    )

plt.show()
