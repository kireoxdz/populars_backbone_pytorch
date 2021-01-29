import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.nasnet import nasnetalarge


''' backbone替换为nasnet '''
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


class NasNetlarge(nn.Module):
    def __init__(self):
        super(NasNetlarge, self).__init__()

        pretrained_model = nasnetalarge(num_classes=1000, pretrained='imagenet')

        self.pre = pretrained_model

    def features(self, input):
        x_conv0 = self.pre.conv0(input)
        x_stem_0 = self.pre.cell_stem_0(x_conv0)  # 168x128x128
        x_stem_1 = self.pre.cell_stem_1(x_conv0, x_stem_0)  # 336x64x64

        x_cell_0 = self.pre.cell_0(x_stem_1, x_stem_0)  # 1008x64x64
        x_cell_1 = self.pre.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.pre.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.pre.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.pre.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.pre.cell_5(x_cell_4, x_cell_3)  # 1008x64x64

        x_reduction_cell_0 = self.pre.reduction_cell_0(x_cell_5, x_cell_4)  # 1344x32x32

        x_cell_6 = self.pre.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.pre.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.pre.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.pre.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.pre.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.pre.cell_11(x_cell_10, x_cell_9)  # 2016x32x32

        x_reduction_cell_1 = self.pre.reduction_cell_1(x_cell_11, x_cell_10)  # 2688x16x16

        x_cell_12 = self.pre.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.pre.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.pre.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.pre.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.pre.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.pre.cell_17(x_cell_16, x_cell_15)  # 4032x16x16
        return x_stem_0, x_cell_5, x_cell_11, x_cell_17

    def forward(self, x):
        c2, c3, c4, c5 = self.features(x)

        return c5

if __name__ == '__main__':
    import torch

    input = torch.randn(1, 3, 512, 512)
    net = NasNetlarge()
    out = net(input)
    print(out.shape)