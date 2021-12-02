import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import random
import data



class To01():
    def __call__(self, tensor:torch.Tensor):
        return tensor.sub_(tensor.min()).div_(tensor.max())

if __name__ == "__main__":
    root_train = "ImageSet/train"
    root_test = "ImageSet/test"
    transform_test = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomRotation((0, 359), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomApply(T.GaussianBlur(), p=.33),
        data.PILToTensor(),
        T.RandomApply(data.RandomPatch, p=0.75),
        To01(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    batch_size_train = 32
    batch_size_test = 128
    trainloader, testloader, trainset, _ = data.get_dataloaders(
        root_train, root_test, transform_train, transform_test, batch_size_train, batch_size_test
    )
    num_classes = len(trainset.classes)
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = torch.nn.Linear(2048, num_classes)

    optimizer = torch.optim.SGD(lr=.001, momentum=.9, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    net.cuda()
