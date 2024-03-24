import json
import string
import random

import numpy as np
import pandas as pd
from keras import Input, Model
from keras.layers import Embedding, LSTM, Dense
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

json_file = open('data.json', 'r')

json_data = json.load(json_file)

type_list = []
question_list = []
answers_list = {}

for intent in json_data['data']:
    answers_list[intent['type']] = intent['answers']  # make a dictionary with key pairs based on the type and then as a value sets an array of answers.
    for lines in intent['questions']: # this cycle iterates over every question and stores it in question_list and add the type of question to the type_list accordingly.
        question_list.append(lines)
        type_list.append(intent['type'])

json_file.close()

data_table = pd.DataFrame({"inputs": question_list, "types": type_list})

# removing unnecessary punctuation for faster training, except question mark
data_table['inputs'] = data_table['inputs'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data_table['inputs'] = data_table['inputs'].apply(lambda wrd: ''.join(wrd))

# processing every word, and converting it from string to integers, kind of like a table with two columns one for the integer value (token, unique) and the other the most used word.
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data_table['inputs'])
train_data = tokenizer.texts_to_sequences(data_table['inputs'])

# makes a number data of type NumPY array, so that every array element is the same length.
padded_train_data = pad_sequences(train_data)

# converting the data of types to numbers for training
le = LabelEncoder()
y_train = le.fit_transform(data_table['types'])

input_shape = padded_train_data.shape[1]

vocabulary = len(tokenizer.word_index)

output_length = le.classes_.shape[0]

inputs = Input(shape=(input_shape,))
output = Embedding(input_dim=vocabulary+1, output_dim=10)(inputs)
output = LSTM(4)(output)
output = Dense(output_length, activation='softmax')(output)
model = Model(inputs=inputs, outputs=output)

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

train = model.fit(padded_train_data,y_train,epochs=55)

while True:
  texts_p = []
  prediction_input = input('You : ')
  #removing punctuation and converting to lowercase
  prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
  prediction_input = ''.join(prediction_input)
  texts_p.append(prediction_input)
  #tokenizing and padding
  prediction_input = tokenizer.texts_to_sequences(texts_p)
  prediction_input = np.array(prediction_input).reshape(-1)
  print(prediction_input)
  prediction_input = pad_sequences([prediction_input],input_shape)
  #getting output from model
  output = model.predict(prediction_input)
  output = output.argmax()
  #finding the right tag and predicting
  response_tag = le.inverse_transform([output])[0]
  print("Chatkata : ", random.choice(answers_list[response_tag]))
  if response_tag == "goodbye":
    break