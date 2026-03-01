import numpy as np
import pandas as pd


class NeuralNetwork:
    def __init__(
        self, n_inputs, layers_structure, activation_type=None, loss_type="MSE"
    ):
        self.loss_type = loss_type
        self.activation_type = (
            activation_type
            if activation_type
            else ["sigmoid" for _ in layers_structure]
        )
        self.w = [
            np.random.randn(
                layers_structure[i], layers_structure[i - 1] if i else n_inputs
            )
            for i in range(len(layers_structure))
        ]
        self.b = [
            np.random.randn(layers_structure[i], 1)
            for i in range(len(layers_structure))
        ]

    def activation_function(self, z, i=0):
        if self.activation_type[i] == "sigmoid":
            return self.sigmoid(z)
        if self.activation_type[i] == "tanh":
            return self.tanh(z)
        if self.activation_type[i] == "ReLU":
            return self.relu(z)
        if self.activation_type[i] == "LReLU":
            return self.lrelu(z)
        return z

    def lrelu(self, z):
        return np.where(z < 0, 0.01 * z, z)

    def relu(self, z):
        return np.maximum(0, z)

    def tanh(self, z):
        ez = np.exp(z)
        e_z = np.exp(-z)
        return (ez - e_z) / (ez + e_z)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def activation_derivative(self, z, i=0):
        if self.activation_type[i] == "sigmoid":
            sigmoid = self.sigmoid(z)
            return sigmoid * (1 - sigmoid)
        if self.activation_type[i] == "tanh":
            return 1 - self.tanh(z) ** 2
        if self.activation_type[i] == "ReLU":
            return np.where(z > 0, 1, 0)
        if self.activation_type[i] == "LReLU":
            return np.where(z > 0, 1, 0.01)
        return 1

    def forward(self, input, structure=None):
        a = input.T
        if structure is not None:
            structure[0]["a"] = input

        for i, layer in enumerate(self.w):
            z = self.w[i] @ a + self.b[i]
            a = self.activation_function(z, i)
            if structure is not None:
                structure[i + 1]["a"] = a
                structure[i + 1]["z"] = z
        return a.T

    def cost(self, input, output):
        return np.mean(self.loss_function(input, output))

    def loss_function(self, input, output):
        y_ = self.forward(input)
        y = output
        if self.loss_type == "MSE":
            return (y - y_) ** 2
        elif self.loss_type == "CE":
            return -(y * np.log(y_) + (1 - y) * np.log(1 - y_))

    def loss_function_derivative(self, a, y):
        if self.loss_type == "MSE":
            return -2 * (y - a)
        if self.loss_type == "CE":
            return -y / a + (1 - y) / (1 - a)

    def fit(self, x, y, r=0.01, epochs=1, splitting_factor=10):
        batch_size = x.shape[0] // splitting_factor
        for i in range(epochs):
            j = 0
            while j < splitting_factor:
                start = j * batch_size
                end = start + batch_size
                batch_x = x[start:end]
                batch_y = y[start:end]
                self.update(batch_x, batch_y, r)
                j += 1
            if end < x.shape[0]:
                start = end
                end = x.shape[0]
                batch_x = x[start:end]
                batch_y = y[start:end]
                self.update(batch_x, batch_y, r)

    def update(self, x, y, r):
        network_structure = [{} for _ in range(len(self.w) + 1)]
        self.forward(x, network_structure)
        cost = self.cost(x, y)
        x, y = x.T, y.T
        a = network_structure[-1]["a"]
        z = network_structure[-1]["z"]

        da = self.loss_function_derivative(a, y)
        dz = self.activation_derivative(z, -1) * da

        db = np.mean(dz, axis=1, keepdims=True)

        pa = network_structure[-2]["a"]
        dw = np.mean(dz @ pa.T, axis=1, keepdims=True)

        self.w[-1] -= r * dw
        self.b[-1] -= r * db


data_size = 1000
data = pd.DataFrame(
    {
        "A": np.random.randint(-10, 10, data_size),
        "B": np.random.randint(-10, 10, data_size),
    }
)

data["sum"] = data["A"] + data["B"]
data["sum_is_positive"] = np.where(data["sum"] > 0, 1, 0)
data["sum_is_pair"] = np.where(data["sum"] % 2 == 0, 1, 0)


x = data[["A", "B"]].values
y = data[["sum_is_positive", "sum_is_pair"]].values


netty = NeuralNetwork(2, (1, 2), ["LReLU", "sigmoid"], "CE")

print(netty.cost(x, y))
netty.fit(x=x, y=y, r=0.0005, epochs=10**4, splitting_factor=1)
print(netty.cost(x, y))

