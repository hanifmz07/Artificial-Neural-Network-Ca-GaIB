import numpy as np
from tqdm import tqdm
import math
from .utils import f1_score


class ANN:
    """ANN implementation for classification task.
    All layers use ReLU activation function except for the output layer. The output layer use sigmoid activation function
    """

    def __init__(self, layer_info, learning_rate=0.05, epochs=50, batch_size=1):
        self.parameters = {}
        self.gradients = {}
        self.activation_values = {}
        self.linear_values = {}
        self.learning_rate = learning_rate
        self.layer_info = layer_info
        self.epochs = epochs
        self.batch_size = batch_size

    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def _relu(self, X):
        return np.maximum(X, 0)

    def _derivative_sigmoid(self, X):
        return self._sigmoid(X) * (1 - self._sigmoid(X))

    def _derivative_relu(self, X):
        return np.where(X > 0, 1, 0)

    def _init_parameters(self):
        n_layers = len(self.layer_info)
        print(n_layers)
        for l in range(1, n_layers):
            self.parameters[f"W{l}"] = (
                np.random.randn(self.layer_info[l], self.layer_info[l - 1]) * 0.01
            )
            self.parameters[f"b{l}"] = np.zeros((self.layer_info[l], 1))

    def _deep_forward_propagation(self, X):
        n_layers = len(self.layer_info)

        A = X.copy()
        self.activation_values[f"A0"] = X.copy()

        for l in range(1, n_layers - 1):
            A_prev = A.copy()

            # Calculate forward prop for 1 layer
            W = self.parameters[f"W{l}"].copy()
            b = self.parameters[f"b{l}"].copy()
            Z = np.dot(W, A_prev) + b
            A = self._relu(Z)

            # Store linear and activation values
            self.linear_values[f"Z{l}"] = Z
            self.activation_values[f"A{l}"] = A

        # Calculate forward prop of the output layer
        W = self.parameters[f"W{n_layers - 1}"].copy()
        b = self.parameters[f"b{n_layers - 1}"].copy()
        Z_out = np.dot(W, A) + b
        A_out = self._sigmoid(Z_out)

        # Store linear and activation values of the output layer
        self.linear_values[f"Z{n_layers - 1}"] = Z_out
        self.activation_values[f"A{n_layers - 1}"] = A_out

        return A_out

    def _compute_cost(self, A_out, Y):
        n = Y.shape[1]
        cost = np.sum((Y - A_out) ** 2) / n
        cost = np.squeeze(cost)
        return cost

    def _deep_backward_propagation(self, Y):
        n_layers = len(self.layer_info)

        # Load variables needed for backprop
        A_out = self.activation_values[f"A{n_layers - 1}"]
        A_old = self.activation_values[f"A{n_layers - 2}"]
        Z_out = self.linear_values[f"Z{n_layers - 1}"]

        # Calculate gradients of output layer
        dA_out = (-2 / Y.shape[1]) * (Y - A_out)
        dZ_out = dA_out * self._derivative_sigmoid(Z_out)
        dW_out = np.dot(A_old, dZ_out.T).T
        db_out = np.sum(dZ_out, axis=1, keepdims=True)

        # Store gradients of output layer
        self.gradients[f"dA{n_layers - 1}"] = dA_out
        self.gradients[f"dZ{n_layers - 1}"] = dZ_out
        self.gradients[f"dW{n_layers - 1}"] = dW_out
        self.gradients[f"db{n_layers - 1}"] = db_out

        for l in reversed(range(1, n_layers - 1)):
            # Load variables needed for backprop
            dZ_old = self.gradients[f"dZ{l + 1}"]
            W_old = self.parameters[f"W{l + 1}"]
            A_old = self.activation_values[f"A{l - 1}"]
            Z = self.linear_values[f"Z{l}"]

            # Calculate gradients
            dA = np.dot(dZ_old.T, W_old).T
            dZ = dA * self._derivative_relu(Z)
            dW = np.dot(A_old, dZ.T).T
            db = np.sum(dZ, axis=1, keepdims=True)

            # Store gradients
            self.gradients[f"dA{l}"] = dA
            self.gradients[f"dZ{l}"] = dZ
            self.gradients[f"dW{l}"] = dW
            self.gradients[f"db{l}"] = db

    def _update_parameters(self):
        n_layers = len(self.layer_info)
        for l in range(1, n_layers):
            self.parameters[f"W{l}"] -= self.learning_rate * self.gradients[f"dW{l}"]
            self.parameters[f"b{l}"] -= self.learning_rate * self.gradients[f"db{l}"]

    def fit(self, X_train, y_train):
        self._init_parameters()
        n_batches = math.ceil(X_train.shape[1] / self.batch_size)
        costs = []
        with tqdm(total=self.epochs) as pbar:
            pbar.set_description_str("Epoch")
            for i in range(1, self.epochs + 1):
                cost_per_epoch = []
                acc_metric_per_epoch = []
                for batch in range(n_batches):
                    lower_bound = self.batch_size * batch
                    upper_bound = min(self.batch_size * (batch + 1), X_train.shape[1])
                    X_batch = X_train[:, lower_bound:upper_bound]
                    y_batch = y_train[:, lower_bound:upper_bound]

                    A_out = self._deep_forward_propagation(X_batch)

                    y_pred = np.where(np.squeeze(A_out.T) >= 0.5, 1, 0)
                    y_true = np.squeeze(y_batch)
                    cost_per_epoch.append(self._compute_cost(A_out, y_batch))
                    acc_metric_per_epoch.append(f1_score(y_true, y_pred))

                    self._deep_backward_propagation(y_batch)

                    self._update_parameters()

                avg_cost = sum(cost_per_epoch) / len(cost_per_epoch)
                avg_f1 = sum(acc_metric_per_epoch) / len(acc_metric_per_epoch)
                costs.append(avg_cost)
                pbar.set_postfix(loss=avg_cost, f1_score=avg_f1)
                pbar.update(1)
        return costs

    def predict(self, X_test):
        A_out = self._deep_forward_propagation(X_test)
        return np.where(np.squeeze(A_out.T) >= 0.5, 1, 0)
