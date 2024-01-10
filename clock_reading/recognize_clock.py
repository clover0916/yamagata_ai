import random
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

IMAGE_SIZE = 112

# モデルのロード
model = load_model("trained_model.keras")

# 画像ファイルのリストを読み込む
data_file = "./clocks_all.txt"
with open(data_file, "r") as file:
    data = file.readlines()

# ランダムに10個の画像を選択
data = random.sample(data, 10)

# 選択した画像に対して予測を行い、誤差を計算
for line in data:
    line = line.strip().split()
    image_path = line[0]
    hour = int(line[1])
    minute = int(line[2])
    second = int(line[3])

    # 画像のロード
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0

    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    # 予測の実行
    predictions = model.predict(image, verbose=0)

    predictions = np.round(predictions).astype(int)
    print(f"Actual time: {hour:02d}:{minute:02d}:{second:02d}")
    print(
        f"Predicted time: {predictions[0][0]:02d}:{predictions[0][1]:02d}:{predictions[0][2]:02d}"
    )
