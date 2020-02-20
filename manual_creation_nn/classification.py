import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from .simple_nn import Graph, add, matmul, Placeholder, Variable, Session, Operation


def sigmoid(z):
    return 1 / (np.exp(-z) + 1)


sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)

# plt.plot(sample_z, sample_a)
# plt.show()


class Sigmoid(Operation):
    def __init__(self, z):
        super(Sigmoid, self).__init__([z])

    def compute(self, z_val):
        super(Sigmoid, self).compute([z_val])
        return sigmoid(z_val)


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
features = data[0]
labels = data[1]

x = np.linspace(0, 11, 10)
y = -x + 5

plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.plot(x, y)
plt.show()


g = Graph()
g.set_as_default()
x = Placeholder()
w = Variable([1, 1])
b = Variable(-5)
z = add(matmul(x, w), b)
a = Sigmoid(z)
sess = Session()
result = sess.run(operation=a, feed_dict={x: [8, 10]})


def get_result():
    return result


# print(result)
