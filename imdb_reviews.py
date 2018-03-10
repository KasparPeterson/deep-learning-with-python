from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np


def decode(sequence):
	word_index = imdb.get_word_index()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	return " ".join(reverse_word_index.get(i - 3, "?") for i in sequence)


def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results


def get_model():
	model = models.Sequential()
	model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
	model.add(layers.Dense(16, activation="relu"))
	model.add(layers.Dense(1, activation="sigmoid"))
	model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
	return model


def get_data():
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
	x_train = vectorize_sequences(train_data)
	x_test = vectorize_sequences(test_data)
	y_train = np.asarray(train_labels).astype("float32")
	y_test = np.asarray(test_labels).astype("float32")
	return x_train[:10000], x_train[10000:], y_train[:10000], y_train[10000:]


x_val, x_train, y_val, y_train = get_data()

model = get_model()
history = model.fit(
	x_train,
	y_train,
	epochs=20,
	batch_size=512,
	validation_data=[x_val, y_val])

history_dict = history.history
print("Dict: ", str(history_dict.keys()))


