import torch
import torch.nn as nn
import torch.nn.functional as F

from  backbone.senet import senet154

# from heatmap_train import load_trained_model


''' backbon使用senet152 '''
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SENet(nn.Module):
    def __init__(self, pretrained=True):
        super(SENet, self).__init__()

        if pretrained:

            ## 可以替换为50等等
            pretrained_model = senet154(num_classes=1000, pretrained='imagenet')  ## 这里更换senet的类型

            self.layer0 = pretrained_model.layer0
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4
        else:
            pretrained_model = senet154(num_classes=1000, pretrained=None)  ## 这里更换senet的类型

            self.layer0 = pretrained_model.layer0
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = pretrained_model.layer4


    def forward(self, x):
        # Bottom-up
        c1 = self.layer0(x)
        c2 = self.layer1(c1)  # ch = 256
        c3 = self.layer2(c2)  # ch = 512
        c4 = self.layer3(c3)  # ch = 1024
        c5 = self.layer4(c4)  # ch = 2048

        return c5

if __name__ == '__main__':

    input = torch.randn(1, 3, 512, 512)
    net = SENet(pretrained=True)
    out = net(input)
    print(out.shape)
