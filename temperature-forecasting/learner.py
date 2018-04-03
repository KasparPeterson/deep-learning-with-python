from matplotlib import pyplot as plt
import numpy as np

FILENAME = "jena_climate_2009_2016.csv"


def get_header_and_lines():
    f = open(FILENAME)
    data = f.read().split("\n")
    f.close()

    return data[0].split(","), data[1:]


def get_float_data(_header, _lines):
    data = np.zeros((len(_lines), len(_header) - 1))
    for i, line in enumerate(_lines):
        values = [float(x) for x in line.split(",")[1:]]
        data[i, :] = values
    return data


def plot_temperatures(_float_data):
    temp = _float_data[:, 1]
    plt.plot(range(len(temp)), temp)
    plt.figure()
    plt.plot(range(1400), temp[:1400])
    plt.show()


header, lines = get_header_and_lines()
float_data = get_float_data(header, lines)
plot_temperatures(float_data)
