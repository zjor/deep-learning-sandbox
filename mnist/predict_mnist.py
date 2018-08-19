import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from scipy.ndimage import imread

def greyscale(rgb):
	return ((0.3 * rgb[0]) + (0.59 * rgb[1]) + (0.11 * rgb[2]))

def load_image_data(filename):
	image = imread(filename)	

	data = []
	for x in range(0, 28):
		row = []
		for y in range(0, 28):
			row.append(greyscale(image[x][y]))
		data.append(row)
	return data

def print_image_as_ascii(image_data):
	for x in range(0, len(image_data)):
		line = []
		for y in range(0, len(image_data[x])):
			line.append('.' if image_data[x][y] < 16 else 'x')
		print ''.join(line)


model_filename = "mnist_model.h5"
model = load_model(model_filename)

image_data = load_image_data(sys.argv[1])
print_image_as_ascii(image_data)

predictions = model.predict(np.array([image_data]).reshape((1, 28 * 28)))[0]
detected_digit = filter(lambda x: x[1] > .5, enumerate(predictions))[0][0]
	
print "Detected: %s" % detected_digit