import os

imdb_dir = "aclImdb"
train_dir = os.path.join(imdb_dir, "train")


def load_data():
    labels = []
    texts = []

    for label_type in ["neg", "pos"]:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == ".txt":
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()

                if label_type == "neg":
                    labels.append(0)
                else:
                    labels.append(1)

    return texts, labels
