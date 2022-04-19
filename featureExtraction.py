import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np


def extrating_features(model, data, return_nodes: list):
    train_nodes, eval_nodes = get_graph_node_names(model)
    feat_ext = create_feature_extractor(model, return_nodes=return_nodes)
    with torch.no_grad():
        i = 0
        for X, y in data:  
            if i == 0:
                all_data = X
                all_labels = y
                i = 1
            else:
                all_data = torch.concat((all_data, X), dim=0) 
                all_labels = torch.concat((all_labels, y), dim=0) 
        out = feat_ext(all_data)
    features_map = []
    for _, layer in enumerate(return_nodes):
        # qui ci sono anche i 3 canali, noi mettiamo tutto insieme
        features_map.append(out[layer].numpy().reshape(out[layer].shape[0], -1))
    return features_map, all_labels
