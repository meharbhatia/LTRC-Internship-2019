from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np, string, re

def processText (textInputs):
    finalList = []
    exclude = set(string.punctuation)
    for textList in textInputs:
        textList = [text.lower().replace(",", " COMMA").replace("'","") for text in textList]
        textList = [''.join(ch for ch in text if ch not in exclude) for text in textList]
        textList = [re.sub("\d","",text) for text in textList]
        finalList.append(textList)
    return finalList

batch_size = 64
epochs = 10
latent_dim = 256
num_samples = 10000

data_path_inp = 'enghin/train.en'
data_path_inp_val = 'enghin/dev.en'
data_path_tar = 'enghin/train.hi'
data_path_tar_val = 'enghin/dev.hi'
data_path_inp_test = 'enghin/test.en'
data_path_tar_test = 'enghin/test.hi'

# Vectorize the data.
# Vectorize the data.
input_texts, target_texts = [], []
input_words, target_words = set(),  set()

with open(data_path_inp, 'r', encoding='utf-8') as f:
    lines_inp = f.read().split('\n')
with open(data_path_inp_val, 'r', encoding='utf-8') as f:
    lines_inp.extend(f.read().split('\n'))
    
with open(data_path_inp_test, 'r', encoding='utf-8') as f:
    lines_inp_test = f.read().split('\n')
with open(data_path_tar_test, 'r', encoding='utf-8') as f:
    lines_tar_test = f.read().split('\n')

with open(data_path_tar, 'r', encoding='utf-8') as f:
    lines_tar = f.read().split('\n')
with open(data_path_tar_val, 'r', encoding='utf-8') as f:
    lines_tar.extend(f.read().split('\n'))

(lines_inp, lines_tar, lines_inp_test, lines_tar_test) = processText([lines_inp, lines_tar, lines_inp_test, lines_tar_test])    
    
for (i,line) in enumerate(lines_inp[:num_samples]):
    input_text = line
    input_texts.append(input_text)
    for word in input_text.split():
        if word not in input_words:
            input_words.add(word)

    target_text = lines_tar[i]
    target_text = 'SOL ' + target_text + ' EOL'
    target_texts.append(target_text)

    for word in target_text.split():
        if word not in target_words:
            target_words.add(word)

for (i,line) in enumerate(lines_inp_test):
    input_text = lines_inp_test[i]
    for word in input_text.split():
        if word not in input_words:
            input_words.add(word)
    
    target_text = lines_tar_test[i]
    for word in target_text.split():
        if word not in target_words:
            target_words.add(word)

max_encoder_seq_length = max([len(txt.split()) for txt in input_texts])
input_words = sorted(list(input_words))
num_src_words = len(input_words)
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_src_words),
    dtype='float32')

max_decoder_seq_length = max([len(txt.split()) for txt in target_texts])
target_words = sorted(list(target_words))
num_targ_words = len(target_words)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_targ_words),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_targ_words),
    dtype='float32')

input_word_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_word_index = dict(
    [(word, i) for i, word in enumerate(target_words)])
reverse_input_word_index = dict(
    (i, word) for word, i in input_word_index.items())
reverse_target_word_index = dict(
    (i, word) for word, i in target_word_index.items())


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t, input_word_index[word]] = 1.
    for t, word in enumerate(target_text.split()):
        decoder_input_data[i, t, target_word_index[word]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_word_index[word]] = 1.

encoder_inputs = Input(shape=(None, num_src_words))
encoder = LSTM(latent_dim, return_state=True)


decoder_inputs = Input(shape=(None, num_targ_words))
decoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=True)

_, stateH, stateC = encoder(encoder_inputs)
encoder_states = [stateH, stateC]


decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_targ_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          epochs=epochs,
          validation_split=0.2,
         batch_size=batch_size)

model.save('SequenceToSequence.h5')

data_path_test_inp = 'enghin/test.en'
data_path_test_arg = 'enghin/test.hi'

input_texts_test = []
target_texts_test = []
    
for (i,line) in enumerate(lines_inp_test):
    input_text = line
    target_text = lines_tar_test[i]
    target_text = 'SOL ' + target_text + ' EOL'
    input_texts_test.append(input_text)
    target_texts_test.append(target_text)

encoder_input_data_test = np.zeros(
    (len(input_texts_test), max_encoder_seq_length, num_src_words),
    dtype='float32')

for (i,line) in enumerate(lines_inp_test):
    for (t, word) in enumerate(line.split()):
        encoder_input_data_test[i, t, input_word_index[word]] = 1.
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

def decode_sequence(input_seq):
    decoded = ''
    states = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_targ_words))
    target_seq[0, 0, target_word_index['SOL']] = 1.
    stop = False
    while not stop:
        output_words, h, c = decoder_model.predict([target_seq] + states)
    
        sampled_word_index = np.argmax(output_words[0, -1, :])
        sampled_word = reverse_target_word_index[sampled_word_index]
        decoded_sentence += sampled_word

        if (sampled_word == 'EOL' or len(decoded_sentence) > max_decoder_seq_length):
            stop = True

        target_seq = np.zeros((1, 1, num_targ_words))
        target_seq[0, 0, sampled_word_index] = 1.

        states = [h, c]
    return decoded_sentence


for seq_index in range(100):
    input_seq = encoder_input_data_test[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
    print ('Actual translation:', target_texts_test[seq_index])