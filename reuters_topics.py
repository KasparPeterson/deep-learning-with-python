from keras import models
from keras import layers
from keras.datasets import reuters
import keras.utils.np_utils as utils
import numpy as np
import plotter


EPOCHS = 9


def vectorize_sequence(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results


def get_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dense(46, activation="softmax"))
	model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
	return model


(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

x_train = vectorize_sequence(train_data)
x_test = vectorize_sequence(test_data)

y_train = utils.to_categorical(train_labels)
y_test = utils.to_categorical(test_labels)

x_val = x_train[:1000]
x_partial_train = x_train[1000:]

y_val = y_train[:1000]
y_partial_train = y_train[1000:]

model = get_model()
history = model.fit(
	x_partial_train,
	y_partial_train,
	epochs=EPOCHS,
	batch_size=512,
	validation_data=(x_val, y_val)
)

history_dict = history.history
plotter.plot_losses(history_dict, EPOCHS)
plotter.clear()
plotter.plot_accuracies(history_dict, EPOCHS)

results = model.evaluate(x_test, y_test)
print("Results: ", results)

predictions = model.predict(x_test)
print("Prediction:", predictions[0])

max_value = np.argmax(predictions[0])
print("Max value: ", max_value)