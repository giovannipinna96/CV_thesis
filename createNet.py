import torch
from torch.nn import ModuleList
import torchvision

def create_network(model, pretrained=True):
    match model:
        case 'resnet50':
            net = torchvision.models.resnet50(pretrained=pretrained)
        case 'resnet18':
            net = torchvision.models.resnet18(pretrained=pretrained)
        case 'resnet101':
            net = torchvision.models.resnet101(pretrained=pretrained)
        case 'vgg16':
            net = torchvision.models.vgg16(pretrained=pretrained)
    
    return net