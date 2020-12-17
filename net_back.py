from torchvision.models import resnet18
import torch.nn as nn
import torch

model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
"""
class res18(nn.Module):
    def __init__(self, num_classes=2):
        super(res18, self).__init__()
        self.base = model
        self.feature = nn.Sequential(
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4          #输出512通道
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  #(batch, 512, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  #(batch, 512, 1, 1)
        self.reduce_layer = nn.Conv2d(1024, 512, 1)
        self.fc  = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        bs = x.shape[0]   #batch size
        x = self.feature(x)    # 输出512通道
        avgpool_x = self.avg_pool(x)   #输出(batch, 512, 1, 1)
        maxpool_x = self.max_pool(x)    #输出(batch,512, 1, 1)
        x = torch.cat([avgpool_x, maxpool_x], dim=1)  #输出(batch, 1024, 1, 1)
        x = self.reduce_layer(x).view(bs, -1)    #输出[batch, 512])
        logits = self.fc(x)    #输出（batch，num_classes)
        return logits
"""
from torchvision.models import resnet34
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
#from models.modelzoo.resnet import resnet18,resnet34,model_urls
# snet50
from splat import SplAtConv2d

model = resnet34(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
'''
定义resnet34
'''

def conv3x3(in_planes, out_planes, stride=1):
    "3*3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class BasicBlock(nn.Module):
    expansion = 1

    # inplanes = channel
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample    # what is the use of downsample ?
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 把shortcut那的channel的维度统一
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)# 因为mnist为（1，28，28）灰度图，因此输入通道数为1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():    # what is it mean?
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        # downsample 主要用来处理H(x)=F(x)+x中F(x)和xchannel维度不匹配问题
        downsample = None

        # self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet34(pretrained=False, **kwargs):
    #Constructs a ResNet-18 model.
    #Args:
        #pretrained (bool): If True, returns a model pre-trained on ImageNet

    # [2, 2, 2, 2] is compared with []*2
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:  # laod pretrained weights
        model.load_state_dict(model_zoo.load_urls(model_urls['resnet34']))
    return model