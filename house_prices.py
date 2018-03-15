from keras import models
from keras import layers
from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt


def get_model():
	model = models.Sequential()
	model.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation="relu"))
	# No activation on the last layer as it is purely linear value
	model.add(layers.Dense(1))
	# mse - mean squared error - good for regression problems
	# mae + mean absolute error - the absolute value of the difference between predictions and targets
	model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
	return model


def smooth_curve(points, factor=0.9):
	smoothed_points = []
	for point in points:
		if smoothed_points:
			previous = smoothed_points[-1]
			smoothed_points.append(previous * factor + point * (1 - factor))
		else:
			smoothed_points.append(point)
	return smoothed_points

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std


# k-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_mae_histories = []


for i in range(k):
	print("Processing fold #", i)
	val_data = train_data[i * num_val_samples: (i +1) * num_val_samples]
	val_targets = train_targets[i * num_val_samples: (i +1) * num_val_samples]

	partial_train_data = np.concatenate(
		[train_data[:i * num_val_samples],
		train_data[(i + 1) * num_val_samples:]],
		axis=0)

	partial_train_targets = np.concatenate(
		[train_targets[:i * num_val_samples],
		train_targets[(i + 1) * num_val_samples:]],
		axis=0)

	print("Num val samples: ", num_val_samples)
	print("Partial train: ", len(partial_train_data))

	model = get_model()
	history = model.fit(
		partial_train_data, 
		partial_train_targets, 
		validation_data=(val_data, val_targets),
		epochs=num_epochs, 
		batch_size=1, 
		verbose=0)

	print(history.history)
	mae_history = history.history["mean_absolute_error"]
	all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel(["Epochs"])
plt.ylabel(["Validation MAE"])
plt.show()

plt.clf()

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel(["Epochs"])
plt.ylabel(["Validation MAE"])
plt.show()

