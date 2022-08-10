from numpy import block
import torchvision
import torch.nn as nn
import torch


def create_network(model: str, pretrained : bool =True, not_freeze=None):
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

def create_dict_resNet50Costum(net, name_file):
    ditc_resnet50custom = {}
    classic_net = torchvision.models.resnet50()
    classic_net.fc = torch.nn.Linear(in_features=2048,
                                    out_features=17,
                                    bias=True
                                    )
    classic_net.load_state_dict(torch.load(name_file, map_location='cpu'))
    for k1, v1 in classic_net.state_dict().items():
        if k1 in net.state_dict().keys():
            ditc_resnet50custom[k1] = v1
    for k2, v2 in net.state_dict().items():
        if k2 not in ditc_resnet50custom.keys():
            ditc_resnet50custom[k2] = v2

    return ditc_resnet50custom, classic_net

class resNet50Costum(torchvision.models.resnet.ResNet): 
    def __init__(self, num_classes):
        super(resNet50Costum, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        del self.fc
        self.fc1 = nn.Linear(2048, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.avgpool(x)
        out = out.reshape(out.shape[0], -1)
        
        out_z = self.fc1(out)
        out_y = self.fc2(out_z)

        return out_z, out_y