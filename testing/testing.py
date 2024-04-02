import json
import random
import string

import keras as kr
import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

json_file = open('data.json', 'r')

json_data = json.load(json_file)

type_list = []
question_list = []
answers_list = {}

for intent in json_data['data']:
    answers_list[intent['type']] = intent['answers']
    for lines in intent['questions']:
        question_list.append(lines)
        type_list.append(intent['type'])

json_file.close()

data_table = pd.DataFrame({"inputs": question_list, "types": type_list})

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data_table['inputs'])
train_data = tokenizer.texts_to_sequences(data_table['inputs'])

padded_train_data = pad_sequences(train_data)

input_shape = padded_train_data.shape[1]

le = LabelEncoder()
y_train = le.fit_transform(data_table['types'])

trained_model = kr.models.load_model('testing/chatkata_trained.keras')

while True:
    texts_p = []
    prediction_input = input('You : ')
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input], input_shape)
    output = trained_model.predict(prediction_input)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    print("Chatkata : ", random.choice(answers_list[response_tag]))
    if response_tag == "goodbye":
        break
