from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt


EPOCHS = 20


def sentence_to_vector(sentence):
	sequence = encode(sentence)
	sequence = preprocess_sequence(sequence)
	return vectorize_sequences([sequence])


def preprocess_sequence(sequence):
	result = []
	for s in sequence:
		if s is not None:
			result.append(s)
	return result


def decode(sequence):
	word_index = imdb.get_word_index()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	return " ".join(reverse_word_index.get(i - 3, "?") for i in sequence)


def encode(sentence):
	word_index = imdb.get_word_index()
	sequence = []
	for word in sentence.split():
		value = word_index.get(word.lower()) + 3
		sequence.append(value)
	return sequence


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
	return x_train, x_test, y_train, y_test


def plot_losses(history_dict):
	loss_values = history_dict["loss"]
	val_loss_values = history_dict["val_loss"]

	epochs = range(1, EPOCHS + 1)

	plt.plot(epochs, loss_values, 'bo', label="Training loss")
	plt.plot(epochs, val_loss_values, "b", label="Validation loss")
	plt.title("Training and validation loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()


def plot_accuracies(history_dict):
	acc_values = history_dict["acc"]
	val_acc_values = history_dict["val_acc"]

	epochs = range(1, EPOCHS + 1)

	plt.plot(epochs, acc_values, "bo", label="Training acc")
	plt.plot(epochs, val_acc_values, "b", label="Validation acc")
	plt.title("Training and validation accuracy")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()

x_train, x_test, y_train, y_test = get_data()

x_val = x_train[:10000]
x_partial = x_train[10000:]
y_val = y_train[:10000]
y_partial = y_train[10000:]

model = get_model()
history = model.fit(
	x_partial,
	y_partial,
	epochs=EPOCHS,
	batch_size=512,
	validation_data=[x_val, y_val])

results = model.evaluate(x_test, y_test)
print("results: ", results)

history_dict = history.history
plot_losses(history_dict)
plt.clf()
plot_accuracies(history_dict)

sentence = "Wow amazing stuff"
vector = sentence_to_vector(sentence)

predicted = model.predict([vector])
print("Predicted:")
print(predicted)

