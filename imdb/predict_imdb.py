import re
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

imdb = tf.keras.datasets.imdb

num_words = 10000

word_index = imdb.get_word_index()

def vectorize_samples(samples):
	result = np.zeros((len(samples), num_words))
	for i, sample in enumerate(samples):
		result[i, sample] = 1.
	return result

def encode_sample(sample):
	sample = re.sub(r"[^\w\s]"," ", sample)
	sample = sample.lower().split()
	return filter(lambda i: i >= 0 and i < num_words, map(lambda w: word_index.get(w, -4) + 3, sample))

sample = sys.argv[1]
print encode_sample(sample)
encoded = vectorize_samples([encode_sample(sample)])
print encoded

model_filename = "imdb_model.h5"
model = load_model(model_filename)

print model.predict(encoded)

