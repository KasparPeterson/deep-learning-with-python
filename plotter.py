import matplotlib.pyplot as plt

def plot_losses(history_dict, epochs):
	loss_values = history_dict["loss"]
	val_loss_values = history_dict["val_loss"]

	epochs_range = range(1, epochs + 1)

	plt.plot(epochs_range, loss_values, 'bo', label="Training loss")
	plt.plot(epochs_range, val_loss_values, "b", label="Validation loss")
	plt.title("Training and validation loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()


def plot_accuracies(history_dict, epochs):
	acc_values = history_dict["acc"]
	val_acc_values = history_dict["val_acc"]

	epochs_range = range(1, epochs + 1)

	plt.plot(epochs_range, acc_values, "bo", label="Training acc")
	plt.plot(epochs_range, val_acc_values, "b", label="Validation acc")
	plt.title("Training and validation accuracy")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()


def clear():
	plt.clf