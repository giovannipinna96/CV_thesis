from re import T
import torch
import transformation
import os
import data
import train
import test
import utils
import featureExtraction
from allParameters import allParameters
import createNet
from lossContrastiveLearning import lossContrastiveLearning
import pandas as pd
from clustering import all_clustering, clustering_methods
import numpy as np
from svm import svm_methods


if __name__ == "__main__":
    # set all parameters
    allParams = allParameters(
        root_train="ImageSet/train",
        root_test="ImageSet/test",
        weights_save_path="models/model.pt",
        batch_size_train=32,
        batch_size_test=128,
        model='resnet50',
        pretrained=True,
        num_epochs=15,
        not_freeze='nothing',
        loss_type='crossEntropy',
        out_net=18,
        is_feature_extraction=True
    )
    # transform the dataset
    transform_train = transformation.get_transform_train()
    transform_test = transformation.get_transform_test()

    # split the dataset
    trainloader, testloader, trainset, _ = data.get_dataloaders(allParams.get_root_train(),
                                                                allParams.get_root_test(),
                                                                transform_train,
                                                                transform_test,
                                                                allParams.get_batch_size_train(),
                                                                allParams.get_batch_size_test()
                                                                )
    #define the number of different classes
    num_classes = len(trainset.classes)

    # import the basic net
    net = createNet.create_network(allParams.get_model(),
                                   allParams.get_pretrained(),
                                   allParams.get_not_freeze()
                                   )
    # define the loss function and set the last part/layer of the network
    if allParams.get_loss_type() == 'crossEntropy':
        loss_fn = torch.nn.CrossEntropyLoss()
        if allParams.get_model() == 'vgg16':
            net.classifier[6] = torch.nn.Linear(in_features=4096,
                                                out_features=allParams.get_out_net(),
                                                bias=True
                                                )
        else:
            net.fc = torch.nn.Linear(in_features=2048,
                                     out_features=allParams.get_out_net(),
                                     bias=True
                                     )
    else:
        loss_fn = lossContrastiveLearning(temperature=0.07)
        if allParams.get_model() == 'vgg16':
            net.classifier[6] = torch.nn.Linear(in_features=4096,
                                                out_features=allParams.get_out_net(),
                                                bias=True
                                                )
        else:
            net.fc = torch.nn.Linear(in_features=2048,
                                     out_features=allParams.get_out_net(),
                                     bias=True
                                     )
    # set optimizer
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=.01,
                                momentum=.9,
                                weight_decay=5e-4
                                )
    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=list(
                                                         range(allParams.get_num_epochs()))[5::2],
                                                     gamma=0.25
                                                     )

    # train
    train.train_model(net,
                      trainloader,
                      loss_fn,
                      optimizer,
                      allParams.get_num_epochs(),
                      lr_scheduler=scheduler,
                      device=allParams.get_device(),
                      loss_type=allParams.get_loss_type()
                      )
    # test
    test.test_model(net,
                    testloader,
                    loss_fn=loss_fn,
                    device=allParams.get_device(),
                    loss_type=allParams.get_loss_type()
                    )


    if allParams.get_is_feature_extraction:
        # extract features
        feat_map, feat_map_labels = featureExtraction.extrating_features(net,
                                                                         testloader,
                                                                         ['layer3', 'layer4']
                                                                         )  # is a numpy array

        # give to each features a cluster
        clusters_obj, y_km, y_fcm_hard, y_fcm_soft, y_ac, y_db = all_clustering(feat_map[1])

        # perform svm with features
        svm_obj = svm_methods()
        svm_obj.create_linear_svm(feat_map[1], feat_map_labels)
        pred = svm_obj.predict_linear_svm(feat_map[1])

        # save the features extraction objects
        utils.save_obj(file_name="pickle_feat_extraction",
                       first=feat_map,
                       second=feat_map_labels,
                       third=clusters_obj,
                       fourth=[y_km, y_fcm_hard, y_fcm_soft, y_ac, y_db],
                       fifth=svm_obj
                       )

    # save all general opbject for reproduce the experiment
    utils.save_obj(file_name="pickle",
                   first=allParams,
                   second=net,
                   third=transform_train,
                   fourth=transform_test,
                   fifth=trainloader,
                   sixth=testloader,
                   seventh=loss_fn,
                   eighth=optimizer,
                   ninth=scheduler
                   )

    # save network weights
    os.makedirs(os.path.dirname(allParams.get_weights_save_path()),
                exist_ok=True
                )
    torch.save(net.state_dict(), allParams.get_weights_save_path())
