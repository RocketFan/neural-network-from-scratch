from operator import index
import numpy as np

activ_funcs = {
    'none': lambda x: x,
    'relu': lambda x: np.maximum(0, x),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x))
}

deriv_funcs = {
    'none': lambda x: np.zeros(x.shape),
    'relu': lambda x: np.where(x <= 0, 0, 1),
    'sigmoid': lambda x: activ_funcs['sigmoid'](x) * (1 - activ_funcs['sigmoid'](x))
}


def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class NeuronsLayer:
    activ_func = None
    deriv_func = None

    def __init__(self, n_inputs, n_neurons, activ_func='none'):
        self.weights = np.random.randn(n_inputs, n_neurons) / n_inputs
        self.biases = np.zeros((1, n_neurons))
        self.activ_func = activ_funcs[activ_func]
        self.deriv_func = deriv_funcs[activ_func]

    def forward(self, inputs):
        self.inputs = inputs
        self.sum_inputs = np.dot(inputs, self.weights) + self.biases
        outputs = self.activ_func(self.sum_inputs)

        return outputs

    def update_params(self, descent_weights, descent_biases):
        self.weights = self.weights - descent_weights
        self.biases = self.biases - descent_biases


class NeuralNetwork:
    def __init__(self, size, funcs='relu', epochs=1000, batch_size=0.5, learning_rate=0.1, seed=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.layers = []
        funcs = ['relu'] * (len(size) - 1) if type(funcs) == str else funcs
        np.random.seed(seed)

        for i in range(len(size) - 1):
            self.layers.append(NeuronsLayer(size[i], size[i + 1], funcs[i]))
        self.layers = np.array(self.layers)

    def predict(self, X, threshold=None):
        output = self.layers[0].forward(X)

        for layer in self.layers[1:]:
            output = layer.forward(output)

        return np.where(output < 0.5, 0, 1) if threshold != None else output

    def backward(self, y, y_pred):
        derivs = (y_pred - y)

        for layer in reversed(self.layers):
            deriv_funcs = layer.deriv_func(layer.sum_inputs)
            derivs = deriv_funcs * derivs
            descent_baises = (derivs.sum(axis=0) * self.lr) / len(y)
            descent_weights = (np.dot(layer.inputs.T, derivs)
                               * self.lr) / len(y)
            layer.update_params(descent_weights, descent_baises)
            derivs = np.dot(derivs, layer.weights.T)

    def fit(self, X, y):
        losses = []
        batch = int(len(y) * self.batch_size)
        # n_baches = int(len(y) / bach)
        # baches = np.array([(i * bach, (i + 1) * bach) if i < (len(y) - 1)
        #           else (i * bach, len(y)) for i in range(n_baches)])

        y = y.reshape((len(y), 1)) if y.ndim == 1 else y

        for epoch in range(self.epochs):
            indexes = np.random.choice(len(y), batch, replace=False)
            X_baches, y_baches = X, y
            y_pred = self.predict(X_baches)
            losses.append([epoch, mse_loss(y_baches, y_pred)])
            self.backward(y_baches, y_pred)

        return np.array(losses)
