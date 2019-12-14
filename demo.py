#!/usr/bin/env python
import numpy as np
from MMIDimReduction import MMINet
import matplotlib.pyplot as plt


# An illustrative example - Two Class: 2D to 1D
x_c1 = np.random.multivariate_normal([-6, 5], [[25, 45], [45, 90]], 200)
x_c2 = np.random.multivariate_normal([5, -7], [[25, 45], [45, 90]], 200)
x_train = np.concatenate((x_c1, x_c2), axis=0)
y_train = np.concatenate((np.zeros(shape=(200, 1), dtype='int64'),
                          np.ones(shape=(200, 1), dtype='int64')))

model = MMINet(input_dim=2, output_dim=1, net='linear')
model.learn(x_train, y_train, num_epochs=10)
z_train = model.reduce(x_train)

# Plot results
plt.subplot(1, 2, 1)
plt.scatter(x_train[y_train[:, 0] == 0, 0], x_train[y_train[:, 0] == 0, 1], c='r')
plt.scatter(x_train[y_train[:, 0] == 1, 0], x_train[y_train[:, 0] == 1, 1], c='b')
plt.axis([-40, 40, -40, 40])
plt.subplot(1, 2, 2)
plt.scatter(z_train[y_train[:, 0] == 0, 0], np.zeros((200, 1)), c='r')
plt.scatter(z_train[y_train[:, 0] == 1, 0], np.zeros((200, 1)), c='b')
plt.axis([-40, 40, -5, 5])
plt.show()
