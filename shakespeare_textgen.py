import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np
import os
import time

# Get the data from the Tensorflow web API
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the file and then decode it into UTF-8
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# The vocabulary consists of unique words, so duplicate words are removed by turning the data structure into a set
vocab = sorted(set(text))

# From my understanding, these layers act like lambda functions. This layer returns the IDs when given a set of words
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab))

# This layer returns words when given a set of IDs. The "invert=True" argument reverses the lookup, enabling the reversed input/output
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)

# chars_from_ids returns multiple arrays with elements of single characters, so reduce_join merges the individual characters into full words
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# This variable stores all the IDs from the text and splits 
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

# The data structure of IDs is then converted into a Tensorflow Dataset
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

# The number of words per sequence
seq_length = 100

# The ID dataset is split up into batches of seq_length+1
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

# Returns a tuple containing the input value and the expected value
def split_input_target(sequence):
    input_text = sequence[:-1] # 'Hello' -> 'Hell'
    target_text = sequence[1:] # 'Hello' -> 'ello'
    return (input_text, target_text)

# Turns each word in the sequences dataset into a pair of a feature (the input) and a label (the expected output)
dataset = sequences.map(split_input_target)

BATCH_SIZE = 64

# For shuffling the dataset
BUFFER_SIZE = 10000

# Make the final dataset for training
dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

# Number of unique words in the dataset
vocab_size = len(vocab)

# Embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

class Shakespeare(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.lstm = LSTM(units=rnn_units, return_sequences=True, return_state=True)
        self.dense = Dense(vocab_size)
    
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)

        if states is None:
            states = self.lstm.get_initial_state(x)

        x, states = self.lstm(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

model = Shakespeare(vocab_size=len(ids_from_chars.get_vocabulary()), embedding_dim=embedding_dim, rnn_units=rnn_units)

