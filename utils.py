import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def extrating_features(model, data, return_nodes=['layer4']):
    train_nodes, eval_nodes = get_graph_node_names(model)
    feat_ext = create_feature_extractor(model, return_nodes=return_nodes)
    with torch.no_grad():
        out = {}
        for X, y in data: #problema salva solo l'ultimo batch
            out = feat_ext(X)
    # generalizzare estrazione features per più layer diversi
    features_map = out['layer4'].numpy().reshape(out['layer4'].shape[0], -1) #qui ci sono anche i 3 canali, noi mettiamo tutto insieme
    return features_map
