import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet
import torch.utils.model_zoo as model_zoo

from models.resnet import ResNet,ResNet_Block
from utils.gaussian_init import Attention_Gate,Gaussian_map

model_urls = {
      'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
      'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
      'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
      'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
      'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
  }


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Down_Sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down_Sample,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        #降采样
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=1, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # print(x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        # print(x.shape)

        return x


class Hour_Glass(nn.Module):
    def __init__(self, layers, use_detect=False):
        self.use_detect = use_detect

        super(Hour_Glass, self).__init__()
        self.resnet_block = ResNet_Block(BasicBlock,layers)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=1)
        self.relu = nn.ReLU()

        self.up0 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        up0 = self.resnet_block(x)
        compress = self.conv1(up0)
        compress = self.relu(compress)

        up1 = self.up0(compress)
        up2 = self.up1(up1)
        # up3 = self.ups[2](up2)

        # print(x.shape,up2.shape)
        if self.use_detect:
            return x + up2, up0
        return x + up2



class Attention_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, attention=False,show=False,attention_init=True):
        self.inplanes = 64
        self.attention = attention
        self.attention_init = attention_init
        self.show = show

        super(Attention_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(16, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        #注意力分支
        self.attention_gate1 = Attention_Gate(in_channel=64, stage=1)
        self.attention_gate2 = Attention_Gate(in_channel=256, stage=2)
        self.attention_gate3 = Attention_Gate(in_channel=512, stage=3)


        #检测分支
        # self.b_conv1 = nn.Conv2d(64,32,kernel_size=3, stride=1,padding=1)
        # self.b_bn1 = nn.BatchNorm2d(32)
        # self.b_relu = nn.ReLU(inplace=True)
        self.down_sample1 = Down_Sample(64,32)
        self.down_sample2 = Down_Sample(32,16)

        self.box_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.BatchNorm2d(16)
        )
        self.fc2 = nn.Linear(256,4)

        self.Map1 = Gaussian_map(256)    #原图为128*128， 初始化为放大四倍
        self.Map2 = self.Map1
        self.Map3 = Gaussian_map(128)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
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

    def show_attention(self,attention_map):
        sns.heatmap(attention_map, cmap="YlGnBu")
        plt.axis('off')
        plt.show()

    def forward(self, x, gts):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)


        b_x = self.down_sample1(x)
        b_x = self.down_sample2(b_x)
        b_x = self.box_branch(b_x)
        b_x = b_x.view(b_x.size(0),-1)
        # print(b_x.shape)
        pred_box = self.fc2(b_x)

        # print("input size: ",x.shape)
        #注意力
        center = [[gt[0] + gt[2] / 2, gt[1] + gt[3] / 2] for gt in gts]
        if self.attention:
            attention_map1 = self.attention_gate1(x)
            #是否需要添加注意力初始化
            if self.attention_init:
                attention_map1 = attention_map1+self.Map1.get_map(center)
            x = x*attention_map1
            if self.show:
                self.show_attention(attention_map1.cpu().detach().numpy()[0,0])

        x = self.layer1(x)
        # print("layer1 size: ", x.shape)

        if self.attention:
            attention_map2 = self.attention_gate2(x,attention_map1)
            if self.attention_init:
                attention_map2 = attention_map2 + self.Map2.get_map(center)
            x = x * attention_map2
            if self.show:
                self.show_attention(attention_map2.cpu().detach().numpy()[0, 0])
        x = self.layer2(x)
        # print("layer2 size: ",x.shape)

        if self.attention:
            attention_map3 = self.attention_gate3(x,attention_map2)
            if self.attention_init:
                attention_map3 += self.Map3.get_map(center)
            x = x * attention_map3
            if self.show:
                self.show_attention(attention_map3.cpu().detach().numpy()[0, 0])
        x = self.layer3(x)
        # print("layer3 size: ",x.shape)

        x = self.layer4(x)
        # print("layer4 size:", x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        pred_cls = self.fc(x)

        return pred_cls, pred_box


class Attention_Net(nn.Module):

    def __init__(self, block=BasicBlock):
        super(Attention_Net, self).__init__()

        self.layer = self._make_layer(block, 256, 2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(256, 256,
                      kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
        )
        self.up_sample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample=None
        layers = []
        layers.append(block(planes, planes, stride, downsample))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.layer(x)
        down = self.downsample(x)

        return x

if __name__ == "__main__":
    # hour_glass = Hour_Glass([2,2,2],use_detect=True)
    # y1,y2 = hour_glass(X)
    # print(y1.shape,y2.shape)
    X = torch.randn((1, 3, 224, 224)).cuda()
    gts = torch.Tensor([[0.5,0.5,0.2,0.2]]).cuda()
    model = Attention_Net()
    print(model)


class Faster_Attention(nn.Module):

    def __init__(self, block, layers, attention=False,show=False,attention_init=True):
        self.inplanes = 256
        self.attention = attention
        self.attention_init = attention_init
        self.show = show

        super(Faster_Attention, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnet_block = ResNet_Block(BasicBlock,layers)
        self.layer4 = self._make_layer(block, 512, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, 2)

        #Hour_Glasses
        self.hour_glass1 = Hour_Glass([2, 2, 2],use_detect=True)
        self.hour_glass2 = Hour_Glass([2, 2, 2],use_detect=False)
        self.inplanes = 256
        self.box_conv = self._make_layer(block, 512, layers[2], stride=2)
        self.box_avg = nn.AvgPool2d(7, stride=1)
        self.box_fc = nn.Linear(512, 4)

        #注意力分支
        self.attention_gate1 = Attention_Gate(in_channel=64, stage=1)
        self.attention_gate2 = Attention_Gate(in_channel=64, stage=2)
        self.attention_gate3 = Attention_Gate(in_channel=64, stage=3)

        self.Map1 = Gaussian_map(56)
        self.Map2 = self.Map1
        self.Map3 = Gaussian_map(56)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
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

    def show_attention(self,attention_map):
        sns.heatmap(attention_map, cmap="rainbow")
        plt.axis('off')
        plt.show()

    def forward(self, x, gts):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x,temp_box = self.hour_glass1(x)
        pred_box = self.box_conv(temp_box)
        pred_box = self.box_avg(pred_box).view(pred_box.shape[0],-1)
        pred_box = self.box_fc(pred_box)

        # print("input size: ",x.shape)
        #注意力
        center = [[gt[0] + gt[2] / 2, gt[1] + gt[3] / 2] for gt in gts]
        if self.attention:
            attention_map1 = self.attention_gate1(x)
            #是否需要添加注意力初始化
            if self.attention_init:
                attention_map1 = attention_map1 + self.Map1.get_map(center)
            #利用残差思想,相当于始终保留之前的特征,却在其上添加了更丰富的特征
            # x = x*attention_map1
            x = x + x * attention_map1
            if self.show:
                self.show_attention(attention_map1.cpu().detach().numpy()[0,0])

        x = self.hour_glass2(x)
        # print("layer1 size: ", x.shape)

        if self.attention:
            # attention_map2 = self.attention_gate2(x)
            attention_map2 = self.attention_gate2(x,attention_map1)
            if self.attention_init:
                attention_map2 = attention_map2 + self.Map2.get_map(center)
            # x = x * attention_map2
            x = x + x * attention_map2
            if self.show:
                self.show_attention(attention_map2.cpu().detach().numpy()[0, 0])

        x = self.resnet_block(x)
        x = self.layer4(x)
        # print("layer4 size:", x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        pred_cls = self.fc(x)

        return pred_cls, pred_box



def resnet18(pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False,num_classes = 6,attention=False,show=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Attention_ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_classes,attention=attention,show=show)
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith("fc")}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

# if __name__ == "__main__":
#     # hour_glass = Hour_Glass([2,2,2],use_detect=True)
#     # y1,y2 = hour_glass(X)
#     # print(y1.shape,y2.shape)
#     X = torch.randn((1, 3, 224, 224)).cuda()
#     gts = torch.Tensor([[0.5,0.5,0.2,0.2]]).cuda()
#     model = Faster_Attention(BasicBlock,[2,2,2,2],attention=True).cuda()
#
#     y = model(X,gts)
#     print(y)

