import torch
from torch.nn import ModuleList
import torchvision
import transformation
import os
import data
import train


if __name__ == "__main__":
    root_train = "ImageSet/train"
    root_test = "ImageSet/test"
    weights_save_path = "models/model.pt"

    transform_test = transformation.get_transform_test()
    transform_train = transformation.get_transform_train()
    batch_size_train = 32
    batch_size_test = 128
    trainloader, testloader, trainset, _ = data.get_dataloaders(
        root_train, root_test, transform_train, transform_test, batch_size_train, batch_size_test
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(trainset.classes)
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = torch.nn.Linear(2048, num_classes)

    num_epochs = 15

    optimizer = torch.optim.SGD(net.parameters(), lr=.01, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(num_epochs))[5::2], gamma=0.25
        )
    loss_fn = torch.nn.CrossEntropyLoss()
    

    train.train_model(net, trainloader, loss_fn, optimizer, num_epochs, lr_scheduler=scheduler, device=device)
    train.test_model(net, testloader, loss_fn=loss_fn, device=device)

    os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)
    torch.save(net.state_dict(), weights_save_path)
