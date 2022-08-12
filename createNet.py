from numpy import block
import torchvision
import torch.nn as nn
import torch


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
    def __init__(self, num_classes, dim_latent:int=32):
        super(resNet50Costum, self).__init__(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        del self.fc
        self.fc1 = nn.Linear(2048, dim_latent)
        self.fc2 = nn.Linear(dim_latent, num_classes)

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
