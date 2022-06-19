import torchvision.datasets as datasets

from opennet import *

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True)

X = mnist_trainset.train_data
y = mnist_trainset.train_labels

# opennet = OpenNetFlat(X.shape[1] * X.shape[2], 6, z_dim=6, iterations=5, display_step=1)  # mnist data image of shape 28 * 28 = 784,
# 0-9 digits recognition = > 10 classes
#opennet = OpenNetCNN([X.shape[1], X.shape[2]], 1, 6, [5, 3], [4, 2], z_dim=6, iterations=5, display_step=1)
opennet = OpenNetResNet([X.shape[1], X.shape[2]], 1, 6,  z_dim=6, iterations=5, display_step=1)
# X = X.reshape(X.shape[0], -1)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y = y.reshape(10000, 6)
opennet.fit(X[:200], y[:200], X_val=X[200:300], y_val=y[200:300])
print('Finish')
