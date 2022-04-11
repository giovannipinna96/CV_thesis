import torch
from torch.nn import ModuleList
import torchvision
import transformation
import os
import data
import train
import test
import utils
from allParameters import allParameters
import createNet
from lossContrastiveLearning import lossContrastiveLearning
import pandas as pd
from clustering import get_clusters


if __name__ == "__main__":
    allParams = allParameters()

    transform_test = transformation.get_transform_test()
    transform_train = transformation.get_transform_train()

    trainloader, testloader, trainset, _ = data.get_dataloaders(
        allParams.get_root_train(), allParams.get_root_test(), transform_train, transform_test,
         allParams.get_batch_size_train(), allParams.get_batch_size_test()
    )
    num_classes = len(trainset.classes)
    net = createNet.create_network(allParams.get_model(), allParams.get_pretrained())
    net.fc = torch.nn.Linear(2048, num_classes)

    optimizer = torch.optim.SGD(net.parameters(), lr=.01, momentum=.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(range(allParams.get_num_epochs()))[5::2], gamma=0.25
        )
    loss_fn = torch.nn.CrossEntropyLoss()

    #freeze layers
    for name, param in net.named_parameters():
        if param.requires_grad and allParams.not_freeze not in name:
             param.requires_grad = False

    train.train_model(net, trainloader, loss_fn, optimizer, allParams.get_num_epochs(), lr_scheduler=scheduler, device=allParams.get_device(), criterion=lossContrastiveLearning(temperature=0.07))
    test.test_model(net, testloader, loss_fn=loss_fn, device=allParams.get_device())

    #extract features
    feat_map = utils.extrating_features(net, testloader) # is a numpy array

    #give to each features a cluster
    d = pd.DataFrame(feat_map[1]) # cluster su solo layer4
    y_cluster_prediction, _, all_distances = get_clusters(d)

    os.makedirs(os.path.dirname(allParams.get_weights_save_path()), exist_ok=True)
    torch.save(net.state_dict(), allParams.get_weights_save_path())
