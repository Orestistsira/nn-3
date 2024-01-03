import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import utils
from rbf_nn import RBFNN

x_train_1, y_train_1 = utils.unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = utils.unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = utils.unpickle("cifar-10/data_batch_3")
x_train_4, y_train_4 = utils.unpickle("cifar-10/data_batch_4")
x_train_5, y_train_5 = utils.unpickle("cifar-10/data_batch_5")

x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5])
y_train = np.concatenate([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

x_test, y_test = utils.unpickle("cifar-10/test_batch")

# Standardize the data (important for RBFN)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

input_dim = 3072
hidden_dim = 20  # Number of RBF neurons
output_dim = 10

gamma = 0.001

rbfnn = RBFNN(input_dim, hidden_dim, output_dim, gamma=gamma)
print('Training...')
rbfnn.fit(x_train, y_train, learning_rate=0.001, epochs=100)

# Make predictions on test data
y_pred = rbfnn.predict(x_test)

# Evaluate the accuracy
accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(f'Test Accuracy: {accuracy:.2f}')
