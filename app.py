import pandas as pd
import tensorflow as tf
import numpy as np

# .csv 형식의 파일을 읽어옴
data = pd.read_csv("gpascore.csv")

# 데이터 전처리하기
data = data.dropna()

x_data = []
for i, rows in data.iterrows():
    x_data.append([rows["gre"], rows["gpa"], rows["rank"]])

y_data = data["admit"].values

""" 딥러닝 모델 디자인하기 """
model = tf.keras.models.Sequential(
    [
        # tf.keras.layers.Dense(2, activation="tanh"),  # hidden layer
        # tf.keras.layers.Dense(4, activation="tanh"),  # hidden layer
        # tf.keras.layers.Dense(8, activation="tanh"),  # hidden layer
        # tf.keras.layers.Dense(16, activation="tanh"),  # hidden layer
        tf.keras.layers.Dense(32, activation="tanh"),  # hidden layer
        tf.keras.layers.Dense(64, activation="tanh"),  # hidden layer
        tf.keras.layers.Dense(128, activation="tanh"),  # hidden layer
        tf.keras.layers.Dense(1, activation="sigmoid"),  # 마지막 출력 레이어
    ]
)

""" 모델 컴파일하기 """
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

""" 모델 학습(fit)시키기 """
# model.fit(학습데이터, 실제정답 , epochs = 10)
model.fit(np.array(x_data), np.array(y_data), epochs=2000)


""" 예측 """
predict = model.predict([[750, 3.70, 3], [400, 2.2, 1]])
print(predict)
