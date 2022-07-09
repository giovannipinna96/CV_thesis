import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn


class resNet50Costum(ResNet):
    def __init__(self, num_classes):
        super(resNet50Costum, self).__init__(BasicBlock, [3, 6, 4, 3], num_classes=num_classes, )
        del self.fc
        self.fc1 = nn.Linear(512, 32)
        self.fc2 = nn.Linear(512, num_classes)

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

        out_z = self.fc1(out)
        out_y = self.fc2(out_z)

        return out_z, out_y



#run
model = resNet50Costum(18)
#model.load_state_dict(models.resnet50(pretrained=True).state_dict())
print('Finish')