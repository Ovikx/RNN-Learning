# A CNN classification project I did before but with RNNs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GRU, LSTM, Input
from tensorflow.keras import Model, callbacks
import numpy as np
from image_preprocessor import ImagePreprocessor
np.set_printoptions(suppress=True)

flavors = {
    0 : 'chocolate',
    1 : 'vanilla'
}
pix = 64

ip = ImagePreprocessor(normalization=255, training_threshold=0.8, color_mode='L')
package = ip.preprocess_dirs(['images/chocolate', 'images/vanilla'], [0, 1], True)

train_features = package['TRAIN_IMAGES']
train_labels = package['TRAIN_LABELS']
test_features = package['TEST_IMAGES']
test_labels = package['TEST_LABELS']

train_ds = tf.data.Dataset.from_tensors((train_features, train_labels)).shuffle(10000)
test_ds = tf.data.Dataset.from_tensors((test_features, test_labels)).shuffle(10000)

model = keras.models.Sequential([
    Input(shape=(pix, pix)),
    LSTM(256),
    Dense(1, activation='sigmoid')
])

print(model.summary())

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss=tf.losses.BinaryCrossentropy(from_logits=False),
    metrics=[keras.metrics.BinaryAccuracy()]
)

model.fit(x=train_ds, validation_data=test_ds, epochs=100)
    
print('EXTERNAL SAMPLE TESTING\nOutputs are [0, 1] where 0 is chocolate and 1 is vanilla\n----------------------')
print(f'Vanilla cupcakes: {model.predict(np.array(ip.file_to_array("images/external_test/v1.jpg")))}')
print(f'Vanilla ice cream: {model.predict(np.array(ip.file_to_array("images/external_test/v2.jpg")))}')
print(f'Chocolate brittle: {model.predict(np.array(ip.file_to_array("images/external_test/c1.jpg")))}')
print(f'Chocolate cookies: {model.predict(np.array(ip.file_to_array("images/external_test/c2.jpg")))}')