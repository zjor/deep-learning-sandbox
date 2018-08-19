import tensorflow as tf
from tensorflow.keras import *

(x_train, y_train),(x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28 * 28))
x_test = x_test.reshape((10000, 28 * 28))
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

model = models.Sequential()
model.add(layers.Dense(512, activation="relu", input_shape=(28 * 28, )))
model.add(layers.Dense(10, activation="softmax"))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              	
model_filename = "mnist_model.h5"

model.fit(x_train, y_train, epochs=5)
model.save(model_filename)

print model.evaluate(x_test, y_test)