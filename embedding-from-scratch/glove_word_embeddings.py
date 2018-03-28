import os
import numpy as np

glove_dir = "glove.6B"


def load_embeddings_index():
    embeddings_index = {}
    f = open(os.path.join(glove_dir, "glove.6B.100d.txt"))
    for line in f:
        values = line.split()
        word = values[0]
        coefficients = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefficients
    f.close()

    print("Found %s word vectors." % len(embeddings_index))
    return embeddings_index
