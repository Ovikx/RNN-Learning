import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()

ds, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_ds, test_ds = ds['train'], ds['test']

print(train_ds.element_spec)

for example, label in train_ds.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())

VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_ds.map(lambda text, label: text))