import torch
from torch.nn import ModuleList
import torchvision

def create_network(model, pretrained=True):
    match model:
        case 'resnet50':
            net = torchvision.models.resnet50(pretrained=pretrained)
            return net

# si può fare per molti altri modelli