import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import allParameters

def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def extrating_features(model, data, return_nodes=['layer3', 'layer4']):
    train_nodes, eval_nodes = get_graph_node_names(model)
    feat_ext = create_feature_extractor(model, return_nodes=return_nodes)
    with torch.no_grad():
        #out = {}
        all_data = torch.zeros(1,3,256,256)
        i = 0
        for X, y in data: #problema salva solo l'ultimo batch
            if i == 0:
                all_data = X
                i = 1
            else:
                all_data = torch.concat((all_data, X), dim=0)
        out = feat_ext(all_data)
    # generalizzare estrazione features per più layer diversi
    features_map = []
    for _, layer in enumerate(return_nodes):
        features_map.append(out[layer].numpy().reshape(out[layer].shape[0], -1)) #qui ci sono anche i 3 canali, noi mettiamo tutto insieme
    return features_map

def load_prestParameters(dict):
    allParams = allParameters(dict.get("root_train"),
        dict.get("root_test"),
        dict.get("weights_save_path"),
        dict.get("batch_size_train"),
        dict.get("batch_size_test"),
        dict.get("device"),dict.get("model"),
        dict.get("pretrained"),
        dict.get("num_epochs"),
        dict.get("not_freeze")
        )
