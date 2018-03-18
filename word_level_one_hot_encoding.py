import numpy as np


def get_indexed_words(samples):
    result = {}
    for sample in samples:
        for word in sample.split():
            if word not in result:
                result[word] = len(result) + 1
    return result


def get_one_hot_encodings(token_index):
    length = len(token_index.values()) + 1
    results = np.zeros(shape=(len(samples), length, length))

    for i, sample in enumerate(samples):
        for j, word in list(enumerate(sample.split()))[:length]:
            index = token_index.get(word)
            results[i, j, index] = 1

    return results


samples = ["The cat sat on the mat.", "The dog ate my homework"]


token_index = get_indexed_words(samples)
one_hot_encoding = get_one_hot_encodings(token_index)

print("Results: ", one_hot_encoding)

