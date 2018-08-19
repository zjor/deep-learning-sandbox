import numpy as np
import tensorflow as tf
from tensorflow.keras import *

imdb = tf.keras.datasets.imdb

num_words = 10000

print "Loading IMDB data..."
(train_data, y_train), (test_data, y_test) = imdb.load_data(num_words=num_words)

word_index = imdb.get_word_index()
rev_word_index = {v: k for k, v in word_index.iteritems()}

def decode_review(review):
	return ' '.join(map(lambda ix: rev_word_index[ix - 3] if ix >= 3 else '?', review))

print "== Review example ==\n\n"
print decode_review(train_data[0])
print "\n"

def vectorize_samples(samples):
	result = np.zeros((len(samples), num_words))
	for i, sample in enumerate(samples):
		result[i, sample] = 1.
	return result

print "Vectorizing data..."
x_train = vectorize_samples(train_data)
x_test = vectorize_samples(test_data)

y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = models.Sequential()

model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train, 
	epochs=4, 
	batch_size=512, 
	validation_data=(x_val, y_val))

model_filename = "imdb_model.h5"
model.save(model_filename)

print model.evaluate(x_test, y_test)
