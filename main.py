from re import T
import argparse
from tkinter import Variable
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
from predictNet import predictNet
import data_triplet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--root_train", type=str, default="ImageSet/train")
    parser.add_argument("--root_test", type=str, default="ImageSet/test")
    parser.add_argument("--loss_type", type=str, default="iiloss")
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--out_net", type=int, default=18)
    parser.add_argument("--is_feature_extraction", type=bool, default=True)
    parser.add_argument("--weights_save_path", type=str, default="models/model.pt")
    parser.add_argument("--pickle_save_path", type=str, default="out")
    parser.add_argument("--is_ml", type=bool, default=True)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument
    args = parser.parse_args()
    # set all parameters
    allParams = allParameters(
        root_train=args.root_train,
        root_test=args.root_test,
        weights_save_path=args.weights_save_path,
        batch_size_train=32,
        batch_size_test=128,
        model='resnet50',
        pretrained=True,
        num_epochs=args.epochs,
        not_freeze='nothing',
        loss_type=args.loss_type,
        out_net=args.out_net,
        is_feature_extraction=args.is_feature_extraction,
        is_ml = args.is_ml,
        optimizer=args.optimizer
    )
    # transform the dataset
    transform_train = transformation.get_transform_train()
    transform_test = transformation.get_transform_test()

    # split the dataset
    if allParams.get_loss_type() == 'triplet':
        trainloader, testloader, trainset, testset = data_triplet.get_dataloaders(allParams.get_root_train(),
                                                                    allParams.get_root_test(),
                                                                    allParams.get_batch_size_train(),
                                                                    allParams.get_batch_size_test(),
                                                                    transform_train,
                                                                    transform_test,
                                                                    lazy=True # ???
                                                                    )  
    elif allParams.get_loss_type() == 'crossEntropy' or allParams.get_loss_type() == 'iiloss':
         trainloader, testloader, trainset, testset = data.get_dataloaders(allParams.get_root_train(),
                                                                    allParams.get_root_test(),
                                                                    transform_train,
                                                                    transform_test,
                                                                    allParams.get_batch_size_train(),
                                                                    allParams.get_batch_size_test()
                                                                    )     
    else:
        trainloader, testloader, trainset, testset = data.get_dataloaders(allParams.get_root_train(),
                                                                    allParams.get_root_test(),
                                                                    utils.TwoCropTransform(transform_train),
                                                                    utils.TwoCropTransform(transform_test),
                                                                    allParams.get_batch_size_train(),
                                                                    allParams.get_batch_size_test()
                                                                    )                                                          
    #define the number of different classes
    num_classes = len(trainset.classes)
    if allParams.get_loss_type() != 'iiloss':
        # import the basic net
        net = createNet.create_network(allParams.get_model(),
                                    allParams.get_pretrained(),
                                    allParams.get_not_freeze()
                                    )
        # define the loss function and set the last part/layer of the network
        if allParams.get_loss_type() == 'crossEntropy':
            loss_fn = torch.nn.CrossEntropyLoss()
        elif allParams.get_loss_type() == 'triplet':
            loss_fn = torch.nn.TripletMarginLoss()
        else:
            loss_fn = lossContrastiveLearning(temperature=args.temperature)

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
        loss_fn = torch.nn.CrossEntropyLoss()
        net = createNet.resNet50Costum(num_classes)

    # set optimizer
    if allParams.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=.01,
                                    momentum=.9,
                                    weight_decay=5e-4
                                    )
    elif allParams.optimizer.lower() == "radam":
        optimizer = torch.optim.RAdam(net.parameters(), lr=.0001)
    else:
        raise NotImplementedError(f"Invalid optimizer {allParams.optimizer}. Please choose from 'sgd' or 'radam'.")
    
    # set scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[50,75],
                                                     gamma=0.1
                                                     )

    # train
    print('Start Train')
    _, _, threshold = train.train_model(net,
                      trainloader,
                      loss_fn,
                      optimizer,
                      allParams.get_num_epochs(),
                      lr_scheduler=scheduler,
                      device=allParams.get_device(),
                      loss_type=allParams.get_loss_type(),
                      num_classes=num_classes
                      )
    # test
    if allParams.get_loss_type() != 'triplet' and allParams.get_loss_type() != 'iiloss':
        print('Start Test')
        test.test_model(net,
                        testloader,
                        loss_fn=loss_fn,
                        device=allParams.get_device(),
                        loss_type=allParams.get_loss_type()
                        )
    else:
        print('Start Test ii loss')
        test.test_model_iiloss(net,
                        testloader,
                        loss_fn=loss_fn,
                        device=allParams.get_device(),
                        threshold=threshold
                        )

    #feat from normal predict
    print('Predict Net')
    feat_predict, feat_predict_leables = predictNet(net, testloader, allParams.get_device())

    try:
        print('Saving pickle_general...')
        utils.save_obj(file_name=f"./{args.pickle_save_path}/pickle_general",
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
        print('Saving pickle_predict Net...')
        utils.save_obj(file_name=f"./{args.pickle_save_path}/pickle_predict_net",
                        first=feat_predict,
                        second=feat_predict_leables
                        )
    except: 
        print('Eccezione salvataggio pickle_cluster_svm')

    # save network weights #TODO check save best 
    print('Saving weithts...')
    os.makedirs(os.path.dirname(allParams.get_weights_save_path()),
                exist_ok=True
                )
    torch.save(net.state_dict(), allParams.get_weights_save_path())
    
    # controllas and it is necessary to extract the features
    if allParams.get_is_feature_extraction:
        # extract features
        print('Extracting features ...')
        feat_map, feat_map_labels = featureExtraction.extrating_features(net, allParams.get_device(),
                                                                         testloader,
                                                                         ['layer1','layer2','layer3', 'layer4']
                                                                         )  # is a numpy array
        try:
            print('Saving pickle_feat_extraction...')
            utils.save_obj(file_name=f"./{args.pickle_save_path}/pickle_feat_extraction",
                            first=feat_map,
                            second=feat_map_labels
                        )
        except: 
            print('Eccezione salvataggio pickle_cluster_svm')

        # delete not used any more variables
        del net
        del transform_train
        del transform_test
        del trainloader
        del testloader
        del loss_fn
        del optimizer
        del scheduler
        del feat_predict
        del feat_predict_leables


        if allParams.is_ml:
            # give to each features a cluster
            list_results_clustering = []
            list_results_svm = []
            print('Start methods ML')
            for i in range(len(feat_map)):
                print(f'clustering {i}')
                #clusters_obj, y_km, y_fcm_hard, y_fcm_soft, y_ac, y_db = all_clustering(feat_map[i])
                cluster_obj = clustering_methods()
                y_km, centers_cluster, all_distances = cluster_obj.kmenas_cluster(feat_map[i])
                y_fcm_hard, y_fcm_soft, fcm_centers = cluster_obj.fuzzy_cluster(feat_map[i])

                #list_results_clustering.append(list([clusters_obj, y_km, y_fcm_hard, y_fcm_soft, y_ac, y_db]))
                list_results_clustering.append(list([cluster_obj,
                                                    y_km, centers_cluster,
                                                    all_distances, y_fcm_hard,
                                                    y_fcm_soft, fcm_centers
                                                    ]))
                
                print(f'linear svm {i}')
                svm_obj = svm_methods()
                svm_obj.create_linear_svm(feat_map[i], feat_map_labels)
                pred = svm_obj.predict_linear_svm(feat_map[i])
                list_results_svm.append(list([svm_obj, pred]))
            
            try:
                # save the features extraction objects
                print('Start saving obj')
                print('Saving pickle clustering and svm...')
                utils.save_obj(file_name=f"./{args.pickle_save_path}/pickle_cluster_svm",
                            first=feat_map,
                            second=feat_map_labels,
                            third=list_results_clustering,
                            fourth=list_results_svm
                            )
            except: 
                print('Eccezione salvataggio pickle_cluster_svm')

    print('Finish')
