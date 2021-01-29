import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

#===========================================================================================
## resnet50
## 主体替换Bottleneck的部件
class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims, height, 1]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, 1, width]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, height, width = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, height, width)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, mhsa=False):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        ### 添加的transformer部件
        if not mhsa:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        else:
            if stride == 2:
                self.conv2 = nn.Sequential(
                    MHSA(planes, width=64, height=32), # for myself
                    nn.AvgPool2d(2, 2)
                )
            else:
                self.conv2 = nn.Sequential(
                    MHSA(planes, width=32, height=16) # for myself -- 是stride==2的一半
                )

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


#===========================================================================================
class ResNetBase(nn.Module):
    def __init__(self, block, num_blocks, pretrained_model=None):
        super(ResNetBase, self).__init__()
        self.in_planes = 64

        if pretrained_model is None:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            # Bottom-up layers
            self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, mhsa=True) ## 仅仅替换最好一层是最小的计算代价
        else:
            self.conv1 = pretrained_model.conv1
            self.bn1 = pretrained_model.bn1
            self.layer1 = pretrained_model.layer1
            self.layer2 = pretrained_model.layer2
            self.layer3 = pretrained_model.layer3
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, mhsa=True)

        # Lateral layers --- resnet50之后
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # # Lateral layers --- resnet18,34
        # self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer4 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride, mhsa=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            ### 修改是否需要mhsa
            layers.append(block(self.in_planes, planes, stride, mhsa))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        ## return F.upsample(x, size=(H,W), mode='bilinear') + y
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) ## 128, 256

        ## 使用的原始输出--encoder
        c2 = self.layer1(c1) ### [b, 256, 128, 256]  ## 若是18则换成[b, 64, 16, 32]
        c3 = self.layer2(c2) ### [b, 512, 64, 128]  ## 若是18则换成[b, 128, 16, 32]
        c4 = self.layer3(c3) ### [b, 1024, 32, 64]  ## 若是18则换成[b, 256, 16, 32]
        c5 = self.layer4(c4) ### [b, 2048, 16, 32]  ## 若是18则换成[b, 512, 16, 32]

        return c5

        # # Top-down -- decoder
        # p5 = self.latlayer1(c5)
        # p4 = self._upsample_add(p5, self.latlayer2(c4))
        # p4 = self.toplayer1(p4)
        #
        # p3 = self._upsample_add(p4, self.latlayer3(c3))
        # p3 = self.toplayer2(p3)
        #
        #
        # c2up = F.upsample(c2, size=(512, 1024), mode='bilinear')
        # p2_in = self._upsample_add(p3, self.latlayer4(c2up))
        # p2 = self.toplayer3(p2_in)
        #
        # return p2, p3, p4, p5


def ResNet50_BoT(pretrained=False):
    model = ResNetBase(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(torch.load('./pth/resnet50.pth'), strict=False)
    return model

if __name__ == '__main__':

    input = torch.randn(1, 3, 512, 1024)
    net = ResNet50_BoT(pretrained=True)

    out = net(input)
    print(out.shape) ## (1, 3, 512, 1024)