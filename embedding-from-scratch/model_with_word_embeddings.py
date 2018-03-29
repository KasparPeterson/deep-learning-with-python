from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np
import data_loader
import glove_word_embeddings
import matplotlib.pyplot as plt

max_len = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
embedding_dim = 100


def get_model():
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(Flatten())
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
    return model


def plot(history):
    acc = history["acc"]
    val_acc = history["val_acc"]
    loss = history["loss"]
    val_loss = history["val_loss"]

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.show()


def get_embedding_matrix():
    embeddings_index = glove_word_embeddings.load_embeddings_index()
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


# texts, labels = data_loader.load_data()
# Loading test data for test results with loading weights
texts, labels = data_loader.load_test_data()
print("Test data loaded, length: ", len(texts))
print("Test data loaded, length: ", len(labels))

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts=texts)
sequences = tokenizer.texts_to_sequences(texts)

#word_index = tokenizer.word_index
#print("Found %s unique tokens." % len(word_index))

#data = pad_sequences(sequences, maxlen=max_len)

#labels = np.asarray(labels)

#print("Shape of data tensor:", data.shape)
#print("Shape of label tensor:", labels.shape)

#indices = np.arange(data.shape[0])
#np.random.shuffle(indices)
#data = data[indices]
#labels = labels[indices]

x_test = pad_sequences(sequences, maxlen=max_len)
y_test = np.asarray(labels)

#x_train = data[:training_samples]
#y_train = labels[:training_samples]

#x_val = data[training_samples: training_samples + validation_samples]
#y_val = labels[training_samples: training_samples + validation_samples]

#embedding_matrix = get_embedding_matrix()
#print("Embedding matrix created")

model = get_model()
model.load_weights("pre_trained_glove_model.h5")
evaluation = model.evaluate(x_test, y_test)
print("Evaluation result:")
print(evaluation)

#model.layers[0].set_weights([embedding_matrix])
#model.layers[0].trainable = False

#print("Train length: ", len(x_train))
#print("Validation length: ", len(x_val))

#history = model.fit(
#    x_train,
#    y_train,
#    epochs=10,
#    batch_size=32,
#    validation_data=(x_val, y_val))
#
#model.save_weights("pre_trained_glove_model.h5")
#plot(history.history)


