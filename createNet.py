import torchvision
import torch


def create_network(model: str, pretrained=True, not_freeze=None):
    """This function allows you to load the network to be used for recognition

    Args:
        model (str): name of the model to import.
                    The name must be in ['resnet50', 'resnet18', 'resnet101','vgg16']
        pretrained (bool, optional): If True, returns a model pre-trained on ImageNet.
                                    Defaults to True.
    Returns:
        torchvision.models: returns the network that is present in torchvision without (avgpool) and the (fc) and with all layer freezed. 
    """
    match model:
        case 'resnet50':
            net = torchvision.models.resnet50(pretrained=pretrained)
        case 'resnet18':
            net = torchvision.models.resnet18(pretrained=pretrained)
        case 'resnet101':
            net = torchvision.models.resnet101(pretrained=pretrained)
        case 'vgg16':
            net = torchvision.models.vgg16(pretrained=pretrained)
            # TODO capire che layer eliminare per rendere generale. Probabilmente avgpool e classifier

    return _not_freeze(_freeze_all(net), not_freeze)


def _freeze_all(net):
    """Freeze all layers.

    Args:
        net (torchvision.models): The network.

    Returns:
        torchvision.models: The network with all layer froze.
    """
    for _, param in net.named_parameters():
        param.requires_grad = False

    return net


def _not_freeze(net, layers: list):
    """Defrost layers.

    Args:
        net (torchvision.models): The network.
        layers (list): list of layers to defrost

    Returns:
        torchvision.models: The network with defrost layer in list layers.
    """
    # freeze layers
    for name, param in net.named_parameters():
        if layers in name:
            param.requires_grad = True

    return net
