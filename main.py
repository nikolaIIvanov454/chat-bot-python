import keras

import json
import numpy as np

json_file = open('data.json', 'r')

json_data = json.load(json_file)['data']

question_list = []
answers_list = []
type_list = []

type_list.append(json_data[0]['type'])
question_list.append(json_data[0]['questions'])

for index in range(0, len(json_data)):
    if index % 2 != 0:
        type_list.append(json_data[index]['type'])
        answers_list.append(json_data[index]['answers'])

print(data)

# print(question_list, answers_list, type_list)

# model = keras.Sequential(
#     [
#         keras.layers.Input(shape=(1, )),
#         keras.layers.Dense(32, activation='relu'),
#         keras.layers.Dense(1)
#     ]
# )

# model.compile(optimizer='adam', loss='mse')

# model.fit(np.array(question_list), np.array(answers_list), epochs=10, batch_size=64)

# result = model.predict(np.array(["What is the sun's color?"], ["What is my name?"]))

# print(result)