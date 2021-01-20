import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nnfs

from neural_network import NeuralNetwork
from nnfs.datasets import spiral_data
from mlxtend.plotting import plot_decision_regions

# np.random.seed(0)
nnfs.init()

X, y = spiral_data(200, 3)
indexes = (y != 2)
X, y = X[indexes], y[indexes]
print(X.max(), X.min())
# y = pd.get_dummies(y).to_numpy()
X = np.array([[1, 0], [1, 0.5], [0.5, 0.5]])
y = np.array([1, 0, 1])

print(X.shape, y.shape)
network = NeuralNetwork([2, 2, 1], 'relu', 0)
losses = network.fit(X, y)
y_pred = network.predict(X)
print(y_pred.shape)

plt.plot(losses[:, 0], losses[:, 1])
plt.show()

fig = plot_decision_regions(X=X, y=y, clf=network)
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
