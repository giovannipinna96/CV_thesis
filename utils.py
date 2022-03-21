import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np

def use_gpu_if_possible():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def extrating_features(model, data, return_nodes=['layer1', 'layer2', 'layer3', 'layer4']):
    train_nodes, eval_nodes = get_graph_node_names(model)
    out = create_feature_extractor(model, return_nodes=return_nodes)
    for i, layer in enumerate(return_nodes):
        #qui sono estratte
        feat_map = out[layer].numpy().squeeze(0)
    return feat_map