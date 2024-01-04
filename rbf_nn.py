import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from history import History


class RBFNN:
    def __init__(self, input_dim, hidden_dim, output_dim, gamma=0.001):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.centers = None
        self.weights = np.random.rand(self.hidden_dim, self.output_dim) - 0.5
        # self.weights = np.random.uniform(-1, 1, size=(self.hidden_dim, self.output_dim))

    def gaussian_rbf(self, x, center):
        return np.exp(-self.gamma * np.linalg.norm(x - center) ** 2)

    def calculate_rbf_layer(self, x):
        rbf_layer = np.zeros((x.shape[0], self.hidden_dim))

        for i in range(self.hidden_dim):
            for j in range(x.shape[0]):
                rbf_layer[j, i] = self.gaussian_rbf(x[j], self.centers[i])

        return rbf_layer

    def kmeans(self, x):
        # Use KMeans to initialize RBF centers
        kmeans = KMeans(n_clusters=self.hidden_dim, n_init='auto')
        kmeans.fit(x)
        return kmeans.cluster_centers_

    def fit(self, x, y, learning_rate=0.01, epochs=100, validation_data=()):
        self.centers = self.kmeans(x)

        # Calculate RBF layer
        rbf_layer_output = self.calculate_rbf_layer(x)

        history = History()
        history.hyperparams = {
            'hid_layers_size': self.hidden_dim,
            'learn_rate': learning_rate,
            'gamma': self.gamma
        }

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            output = np.dot(rbf_layer_output, self.weights)
            output = self.softmax(output)

            # Backward pass
            error = output - y
            gradient = np.dot(rbf_layer_output.T, error)

            # Update weights
            self.weights -= learning_rate * gradient

            # Calculate cross-entropy loss
            loss = np.mean(-np.sum(y * np.log(output), axis=1))
            history.loss_history.append(loss)

            # Calculate accuracy
            train_acc = accuracy_score(np.argmax(y, axis=1), np.argmax(output, axis=1))
            history.train_acc_history.append(train_acc)

            test_acc = 0
            if validation_data:
                val_output = self.predict(validation_data[0])
                test_acc = accuracy_score(np.argmax(validation_data[1], axis=1), np.argmax(val_output, axis=1))
                history.test_acc_history.append(test_acc)

            # Print epoch stats
            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}, "
                  f"Loss: {loss:.2f}")

        history.plot_training_history()

    def predict(self, x):
        rbf_layer_output = self.calculate_rbf_layer(x)
        hidden_output = np.dot(rbf_layer_output, self.weights)
        output = self.softmax(hidden_output)
        return output

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
