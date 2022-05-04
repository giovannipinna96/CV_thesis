import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np


def extrating_features(model, device, data, return_nodes: list):
    """Function that extract the tensors of the intermediate layers of the network.
    The intermediate layers of the network to be considered for extraction are specified
     in the input parameter return_nodes.

    Args:
        model (_type_): Network model
        data (_type_): data from where the features will be extracted
        return_nodes (list): List of layers from which to extract features

    Returns:
        _type_: List of lists of tensors where each element of the list is a layer specified in the input return_nodes
        _type_: list of leables associated with the extracted features
    """
    #train_nodes, eval_nodes = get_graph_node_names(model)
    feat_ext = create_feature_extractor(model, return_nodes=return_nodes)
    with torch.no_grad():
        i = 0
        for X, y in data:
            X = X.to(device)
            y = y.to(device)  
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
