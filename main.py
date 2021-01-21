import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nnfs

from neural_network import NeuralNetwork
from nnfs.datasets import spiral_data
from mlxtend.plotting import plot_decision_regions
from sklearn import datasets

# np.random.seed(0)
nnfs.init()

X, y = spiral_data(400, 2)
print(X.max(), X.min())
# data = datasets.make_moons(n_samples=1000, noise=0.1)
# data = datasets.make_circles(n_samples=500, noise=0.025)
# X = data[0]
# y = data[1]

network = NeuralNetwork([2, 6, 8, 6, 4, 2, 1], 'relu', seed=42)
losses = network.fit(X, y, 1000, batch_size=50, learning_rate=0.03)
y_pred = network.predict(X)
print(y_pred)

plt.plot(losses[:, 0], losses[:, 1])
plt.show()

fig = plot_decision_regions(X=X, y=y, clf=network)
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
