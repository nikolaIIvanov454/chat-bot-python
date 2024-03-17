import json
import string

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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
data_table['inputs'] = data_table['inputs'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation or ltrs == '?'])
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

print(data_table)

# flat_input_data = [item for sublist in question_list for item in sublist]
# flat_output_data = [item for sublist in type_list for item in sublist]
#
# data_frame = pd.DataFrame({ "Input Data": flat_input_data, "Type": flat_output_data })

# print(data_frame)

# for i in range(0, len(answers_list)):
#     print(i)
#
# for i in range(0, len(type_list)):
#     print(i)
# data = pd.DataFrame({ "input": question_list, "output": answers_list })


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
