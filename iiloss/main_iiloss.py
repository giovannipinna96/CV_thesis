import glob
import torch.nn.functional as F
import torch

import transformation
from iilossUtils import loadImages, shuff
from opennet import *

transform_train = transformation.get_transform_train()
transform_test = transformation.get_transform_test()

# path to the directories
pathTrain = r"C:/Users/39320/Desktop/prova/CV_thesis/ImageSet/train"
pathTest = r"C:/Users/39320/Desktop/prova/CV_thesis/ImageSet/test"

labels = [os.path.basename(i) for i in glob.glob(pathTrain + '/*', recursive=True)]

xTrain, yTrain = loadImages(pathTrain, labels, dimension=256)
xTest, yTest = loadImages(pathTest, labels, dimension=256)

#xTrain = xTrain[:300]
#xTest = xTest[:300]
y_train = torch.from_numpy(yTrain)
y_test = torch.from_numpy(yTest)

train_temp = []
for x in xTrain:
    train_temp.append(transform_train(x))

test_temp = []
for x in xTest:
    test_temp.append(transform_test(x))

X_train = torch.stack(train_temp, dim=0)
X_train = X_train.permute(0, 2, 3, 1)
X_test = torch.stack(test_temp, dim=0)
X_test = X_test.permute(0, 2, 3, 1)

X_train, y_train = shuff(X_train, y_train)
X_test, y_test = shuff(X_test, y_test)

X_train = torch.tensor(X_train.numpy())
y_train = torch.tensor(y_train.numpy())
X_test = torch.tensor(X_test.numpy())
y_test = torch.tensor(y_test.numpy())

#X_train = X_train[:2480]
y_train = F.one_hot(y_train.long(), 18)
y_test = F.one_hot(y_test.long(), 18)

# opennetFlat = OpenNetFlat(X_train.shape[1] * X_train.shape[2] * X_train.shape[3], 10, z_dim=6, iterations=5, display_step=1)
# opennet = OpenNetCNN([X_train.shape[1], X_train.shape[2]], X_train.shape[3], 10, [5, 3], [4, 2], z_dim=6, iterations=5,display_step=1)
opennet = OpenNetResNet([X_train.shape[1], X_train.shape[2]], X_train.shape[3], y_train.shape[1], z_dim=6, iterations=2,
                        display_step=1)

# opennetFlat.fit(X_train[:200].reshape(X_train[:200].shape[0], -1), y_train[:200], X_val=X_train[200:300].reshape(X_train[200:300].shape[0], -1), y_val=y_train[200:300])
opennet.fit(X_test, y_test, X_val=X_test, y_val=y_test)

print('Finish')
