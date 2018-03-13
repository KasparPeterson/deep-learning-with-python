from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np
import plotter


EPOCHS = 5


def sentence_to_vector(sentence, word_index):
	sequence = encode(sentence, word_index)
	sequence = preprocess_sequence(sequence)
	return vectorize_sequences([sequence])


def preprocess_sequence(sequence):
	result = []
	for s in sequence:
		if s is not None:
			result.append(s)
	return result


def decode(sequence, word_index):
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	return " ".join(reverse_word_index.get(i - 3, "?") for i in sequence)


def encode(sentence, word_index):
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


def get_data():
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
	x_train = vectorize_sequences(train_data)
	x_test = vectorize_sequences(test_data)
	y_train = np.asarray(train_labels).astype("float32")
	y_test = np.asarray(test_labels).astype("float32")
	return x_train, x_test, y_train, y_test


def get_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
	model.add(layers.Dense(128, activation="relu"))
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dense(32, activation="relu"))
	model.add(layers.Dense(16, activation="relu"))
	model.add(layers.Dense(8, activation="relu"))
	model.add(layers.Dense(1, activation="sigmoid"))
	model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy'])
	return model

x_train, x_test, y_train, y_test = get_data()

x_val = x_test[:10000]
x_test_partial = x_test[10000:]
y_val = y_test[:10000]
y_test_partial = y_test[10000:]

model = get_model()
history = model.fit(
	x_train,
	y_train,
	epochs=EPOCHS,
	batch_size=512,
	validation_data=[x_val, y_val])

results = model.evaluate(x_test_partial, y_test_partial)
print("Results: ", results)

history_dict = history.history
plotter.plot_losses(history_dict, EPOCHS)
plotter.clear()
plotter.plot_accuracies(history_dict, EPOCHS)

sentence = "Wow amazing stuff"
word_index = imdb.get_word_index()
vector = sentence_to_vector(sentence, word_index)

predicted = model.predict([vector])
print("Predicted:", predicted)

"""
TODO:
	* less or more hidden layers
	* fewer or more hidden units
	* mse loss function

	* 3 * 16 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=40
		Results:  [1.427087994016409, 0.84024]
		Predicted: [[0.9084877]]
	* 3 * 16 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=20
		Results:  [0.8680351651114225, 0.85108]
		Predicted: [[0.7841977]]
	* 3 * 16 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=10
		Results:  [0.46989694025993345, 0.86372]
		Predicted: [[0.6608456]]
	* 3 * 16 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=5
		Results:  [0.3451024866771698, 0.8646]
		Predicted: [[0.59942013]]

	* 2 * 32 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=40	
		Results:  [1.284982941285968, 0.8452]
		Predicted: [[0.92397213]]
	* 2 * 32 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=20	
		Results:  [0.8384150679278374, 0.85208]
		Predicted: [[0.7801898]]
	* 2 * 32 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=10	
		Results:  [0.4808311348581314, 0.86172]
		Predicted: [[0.6647064]]

	* 2 * 16 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=20
		results:  [0.7687217784357071, 0.84992]

	* 2 * 8 layers epochs=5
		Results:  [0.29632679869651796, 0.87996]
		Predicted: [[0.569483]]

	* 1 * 16 layers (optimizer="rmsprop", loss="binary_crossentropy") epochs=20
		results:  [0.5207954036951065, 0.85616]

	Custom
	* 64, 32, 16 epochs=10
		Results:  [0.35028558655261993, 0.87152]
		Predicted: [[0.6139845]]
	* 64, 32, 16 epochs=5
		Results:  [0.35028558655261993, 0.87152]
		Predicted: [[0.6139845]]

	* 64. 16 layers epochs=10
		Results:  [0.49013313101291656, 0.86076]
		Predicted: [[0.6762914]]
	* 64, 16 layers epochs=5
		Results:  [0.3309839700603485, 0.87228]
		Predicted: [[0.60446864]]
	* 32, 16 layers epochs=10
		Results:  [0.6152230663967132, 0.83536]
		Predicted: [[0.63469917]]

	* 8, 4, 8 layers epochs=5
		Results:  [0.2916112041091919, 0.88496]
		Predicted: [[0.5929572]]
	* 8, 4, 4 layers epochs=5
		Results:  [0.2949486595916748, 0.88436]
		Predicted: [[0.5807504]]

"""

