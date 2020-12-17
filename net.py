from torchvision.models import resnet34
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
#from models.modelzoo.resnet import resnet18,resnet34,model_urls
# resnet50
from splat import SplAtConv2d

model = resnet34(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
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



def resnet18(pretrained=False, **kwargs):
    #Constructs a ResNet-18 model.
    #Args:
        #pretrained (bool): If True, returns a model pre-trained on ImageNet

    # [2, 2, 2, 2] is compared with []*2
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:  # laod pretrained weights
        model.load_state_dict(model_zoo.load_urls(model_urls['resnet34']))
    return model

def resnet34(pretrained=False, **kwargs):
    #Constructs a ResNet-18 model.
    #Args:
        #pretrained (bool): If True, returns a model pre-trained on ImageNet

    # [2, 2, 2, 2] is compared with []*2
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:  # laod pretrained weights
        model.load_state_dict(model_zoo.load_urls(model_urls['resnet34']))
    return model

def resnet50(pretrained=False, **kwargs):
    #Constructs a ResNet-18 model.
    #Args:
        #pretrained (bool): If True, returns a model pre-trained on ImageNet

    # [2, 2, 2, 2] is compared with []*2
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:  # laod pretrained weights
        model.load_state_dict(model_zoo.load_urls(model_urls['resnet34']))
    return model

def resnet101(pretrained=False, **kwargs):
    #Constructs a ResNet-18 model.
    #Args:
        #pretrained (bool): If True, returns a model pre-trained on ImageNet

    # [2, 2, 2, 2] is compared with []*2
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:  # laod pretrained weights
        model.load_state_dict(model_zoo.load_urls(model_urls['resnet34']))
    return model

def resnet152(pretrained=False, **kwargs):
    #Constructs a ResNet-18 model.
    #Args:
        #pretrained (bool): If True, returns a model pre-trained on ImageNet

    # [2, 2, 2, 2] is compared with []*2
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:  # laod pretrained weights
        model.load_state_dict(model_zoo.load_urls(model_urls['resnet34']))
    return model