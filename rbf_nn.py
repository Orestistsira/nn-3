import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

import utils


class RBFNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.centers = None
        self.weights = np.random.rand(self.hidden_dim, self.output_dim)

    def gaussian_rbf(self, x, center, sigma=1.0):
        return np.exp(-np.linalg.norm(x - center) ** 2 / (2 * sigma ** 2))

    def calculate_rbf_layer(self, x):
        rbf_layer = np.zeros((x.shape[0], self.hidden_dim))

        for i in range(self.hidden_dim):
            for j in range(x.shape[0]):
                rbf_layer[j, i] = self.gaussian_rbf(x[j], self.centers[i])

        return rbf_layer

    def fit(self, x, y, num_clusters=10, learning_rate=0.01, epochs=100):
        # Use KMeans to initialize RBF centers
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
        kmeans.fit(x)
        self.centers = kmeans.cluster_centers_

        # Calculate RBF layer
        rbf_layer = self.calculate_rbf_layer(x)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            hidden_output = np.dot(rbf_layer, self.weights)
            output = self.softmax(hidden_output)

            # Backward pass
            error = output - y
            gradient = np.dot(rbf_layer.T, error)

            # Update weights
            self.weights -= learning_rate * gradient

            # Print loss for monitoring
            loss = np.mean(-np.sum(y * np.log(output), axis=1))
            # Calculate accuracy
            accuracy = accuracy_score(np.argmax(y, axis=1), np.argmax(output, axis=1))
            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {accuracy}, Loss: {loss}")

    def predict(self, x):
        rbf_layer = self.calculate_rbf_layer(x)
        hidden_output = np.dot(rbf_layer, self.weights)
        output = self.softmax(hidden_output)
        return output

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


x_train_1, y_train_1 = utils.unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = utils.unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = utils.unpickle("cifar-10/data_batch_3")
x_train_4, y_train_4 = utils.unpickle("cifar-10/data_batch_4")
x_train_5, y_train_5 = utils.unpickle("cifar-10/data_batch_5")

x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5])
y_train = np.concatenate([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

x_test, y_test = utils.unpickle("cifar-10/test_batch")

input_dim = 3072
hidden_dim = 10  # Number of RBF neurons
output_dim = 10

rbfnn = RBFNN(input_dim, hidden_dim, output_dim)
print('Training...')
rbfnn.fit(x_train, y_train, learning_rate=0.01, epochs=100)

# # Make predictions on test data
# y_pred = rbfnn.predict(x_test)
#
# # Evaluate the accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Test Accuracy: {accuracy:.2f}')
