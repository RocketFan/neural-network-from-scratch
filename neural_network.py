from sklearn.utils import shuffle
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
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
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
    def __init__(self, size, funcs='relu', seed=42):
        self.layers = []
        funcs = [funcs] * (len(size) - 1) if type(funcs) == str else funcs
        np.random.seed(seed)

        for i in range(len(size) - 1):
            self.layers.append(NeuronsLayer(size[i], size[i + 1], funcs[i]))
        self.layers = np.array(self.layers)

    def predict(self, X, threshold=0.5):
        output = self.layers[0].forward(X)

        for layer in self.layers[1:]:
            output = layer.forward(output)

        return np.where(output < 0.5, 0, 1) if threshold != None else output

    def backward(self, y, y_pred, lr):
        derivs = (y_pred - y) * 2

        for layer in reversed(self.layers):
            deriv_funcs = layer.deriv_func(layer.sum_inputs)
            derivs = deriv_funcs * derivs
            descent_baises = (derivs.sum(axis=0) * lr) / len(y)
            descent_weights = (np.dot(layer.inputs.T, derivs)
                               * lr) / len(y)
            derivs = np.dot(derivs, layer.weights.T)
            layer.update_params(descent_weights, descent_baises)

    def fit(self, X, y, epochs=100, batch_size=16, learning_rate=0.4):
        losses = []

        y = y.reshape((len(y), 1)) if y.ndim == 1 else y

        for epoch in range(epochs):
            if X.shape[0] % batch_size == 0:
                n_batches = int(X.shape[0] / batch_size)
            else:
                n_batches = int(X.shape[0] / batch_size ) - 1

            X, y = shuffle(X, y)

            X_batches = [X[batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]
            y_batches = [y[batch_size*i:batch_size*(i+1)] for i in range(0, n_batches)]

            batches_losse = 0

            for i, (X_batch, y_batch) in enumerate(zip(X_batches, y_batches)):
                y_pred = self.predict(X_batch, threshold=None)
                batches_losse += mse_loss(y_batch, y_pred)
                self.backward(y_batch, y_pred, learning_rate)
            
            losses.append([epoch, batches_losse / n_batches])

        return np.array(losses)
