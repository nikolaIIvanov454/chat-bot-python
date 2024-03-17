import tensorflow as tf
import tensorflow_text as tf_text

import keras
import keras.preprocessing.text as keras_text

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
        keras.layers.Input(shape=(1, ), dtype=tf.string),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ]
)

model.compile(optimizer='adam', loss='mse')

tokenizer = keras_text.Tokenizer()
tokenizer.fit_on_texts(question_list)
question_sequences = tokenizer.texts_to_sequences(question_list)

model.fit(np.array(question_sequences), np.array(answers_list), epochs=10, batch_size=64)

input_sequences = tokenizer.texts_to_sequences(input_texts)
result = model.predict(np.array(["What is the sun's color?"], ["What is my name?"]))

print(result)
