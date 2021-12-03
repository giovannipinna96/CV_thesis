import torch
from torch.nn import ModuleList
import torchvision
from torchvision import transforms as T
import os
import data
import train




if __name__ == "__main__":
    root_train = "ImageSet/train"
    root_test = "ImageSet/test"
    weights_save_path = "models/model.pt"

    transform_test = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomRotation((0, 359), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomApply(
            ModuleList([T.GaussianBlur(kernel_size=5)]),
            p=.33),
        data.PILToTensor(),
        T.RandomApply(
            ModuleList([data.RandomPatch(50, 200, [[0,0,0],[100,100,100]], .5)]),
            p=0.75),
        data.To01(),
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

    num_epochs = 15

    optimizer = torch.optim.SGD(net.parameters(), lr=.01, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(range(num_epochs))[5::2], gamma=0.25)
    loss_fn = torch.nn.CrossEntropyLoss()
    

    train.train_model(net, trainloader, loss_fn, optimizer, num_epochs, lr_scheduler=scheduler, device="cuda:0")
    train.test_model(net, testloader, loss_fn=loss_fn, device="cuda:0")

    os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)
    torch.save(net.state_dict(), weights_save_path)

