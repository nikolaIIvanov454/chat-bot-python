import keras

import json
import numpy as np

json_file = open('data.json', 'r')

json_data = json.load(json_file)

question_list = []

answers_list = []

for index in range(0, len(json_data)):
    question_list.append(json_data[index]['question'])
    answers_list.append(json_data[index]['answer'])

model = keras.Sequential(
    [
        keras.layers.Input(shape=(1, )),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ]
)

model.compile(optimizer='adam', loss='mse')

model.fit(np.array(question_list), np.array(answers_list), epochs=10, batch_size=64)

result = model.predict(np.array(["What is the sun's color?"], ["What is my name?"]))

print(result)
