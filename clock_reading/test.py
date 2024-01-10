import random
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

IMAGE_SIZE = 112

# モデルのロード
model = load_model("trained_model.keras")

hour = "9"
minute = "9"
second = "57"

# 画像のロード
image = cv2.imread("test.jpg")
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = image / 255.0

image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

# 予測の実行
predictions = model.predict(image, verbose=0)

predictions = np.round(predictions).astype(int)
print(f"Actual time: {hour}:{minute}:{second}")
print(f"Predicted time: {predictions[0][0]}:{predictions[0][1]}:{predictions[0][2]}")
