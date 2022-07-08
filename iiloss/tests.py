import torch

import data
import transformation
from opennet import *

transform_train = transformation.get_transform_train()
transform_test = transformation.get_transform_test()

trainloader, testloader, trainset, testset = data.get_dataloaders(
    "C:/Users/39320/Desktop/prova/CV_thesis/ImageSet/train",
    "C:/Users/39320/Desktop/prova/CV_thesis/ImageSet/test",
    transform_train,
    transform_test,
    128,
    32
)

X_train = torch.Tensor(2488, 224, 224)
X_test = torch.Tensor(630, 224, 224)
y_train = torch.Tensor(2488, 1)
y_test = torch.Tensor(630, 1)

for X, y in trainloader:
    torch.stack(X, out=X_train, dim=0)
    torch.stack(y, out=y_train, dim=0)

for X, y in testloader:
    torch.stack(X, out=X_test, dim=0)
    torch.stack(y, out=y_test, dim=0)

# opennet = OpenNetFlat(X.shape[1] * X.shape[2], 6, z_dim=6, iterations=5, display_step=1)
# opennet = OpenNetCNN([X.shape[1], X.shape[2]], 1, 6, [5, 3], [4, 2], z_dim=6, iterations=5, display_step=1)
opennet = OpenNetResNet([X_train.shape[1], X_train.shape[2]], x_ch=X_train.shape[3], y_dim=y_train.shape[1], z_dim=6,
                        iterations=5, display_step=1)

opennet.fit(X_train[:200], y_train[:200], X_val=X_train[200:300], y_val=y_train[200:300])

print('Finish')

