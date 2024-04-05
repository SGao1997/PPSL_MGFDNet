import torch
from torchvision import models
from collections import OrderedDict
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.nn.modules import (
    BatchNorm2d,
    Conv2d,
    Identity,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
    UpsamplingBilinear2d,
)
from torchvision.models import resnet
from torchvision.ops import FeaturePyramidNetwork as FPN
import torch.nn as nn
from typing import cast

class changeBackbone(nn.Module):
    def __init__(self, 
                backbone = 'resnet18'):
        super(changeBackbone, self).__init__()

        if backbone in ["resnet18", "resnet34"]:
            max_channels = 512
        elif backbone in ["resnet50", "resnet101"]:
            max_channels = 2048
        else:
            raise ValueError(f"unknown backbone: {backbone}.")

        model_fn = getattr(models, backbone)
        self.backbone = model_fn(pretrained=True)
        self.fpn = FPN(
            in_channels_list=[max_channels // (2 ** (3 - i)) for i in range(4)],
            out_channels=256,
        )

    def forward(self, input):
        x = self.backbone.conv1(input)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        features = [c2, c3, c4, c5]

        fpn_features = self.fpn(
            OrderedDict({f"c{i + 2}": features[i] for i in range(4)})
        )

        features = [v for k, v in fpn_features.items()]

        return cast(Tensor, features)

def Conv3x3ReLUBNs(in_channels,
                   inner_channels,
                   num_convs):

    layers = [nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, 3, 1, 1),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(True),
        nn.Dropout()
    )]
    layers += [nn.Sequential(
        nn.Conv2d(inner_channels, inner_channels, 3, 1, 1),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(True),
        nn.Dropout()    
    ) for _ in range(num_convs - 1)]
    return nn.Sequential(*layers)

class metric_attention(nn.Module):
    def __init__(self, in_channels=256, inner_channels=256, num_convs=2):
        super(metric_attention, self).__init__()
        self.compare = Conv3x3ReLUBNs(in_channels=in_channels, inner_channels=inner_channels, num_convs=num_convs)

        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1), 
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
            nn.Conv2d(in_channels//4, in_channels//4, kernel_size=1, stride=1), 
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        change_coarse = self.compare(torch.abs(x1-x2))
        metric = torch.norm(F.normalize(self.squeeze(x1), dim=1, eps=1e-6)-F.normalize(self.squeeze(x2), dim=1, eps=1e-6), dim=1, keepdim=True)
        return change_coarse * self.sigmoid(metric), metric

class featureFused(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(featureFused, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels*4, out_channels=outchannels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(outchannels),
            nn.ReLU()
        )
    
    def forward(self, input):
        _, _, h, w = input[0].shape
        x2 = input[0]
        x3 = nn.functional.interpolate(input[1], size=(h, w),mode='bilinear', align_corners=True)
        x4 = nn.functional.interpolate(input[2], size=(h, w),mode='bilinear', align_corners=True)
        x5 = nn.functional.interpolate(input[3], size=(h, w),mode='bilinear', align_corners=True)
        out = self.conv(torch.cat((x2, x3, x4, x5), dim=1))

        return out

class buildingTask(nn.Module):
    def __init__(self, inchannels):
        super(buildingTask, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out = self.conv(input[0])

        return out

class changeNet(nn.Module):
    def __init__(self, backbone = 'resnet18'):
        super(changeNet, self).__init__()
        self.backbone = changeBackbone(backbone)

        # get change feature
        self.compare_c2 = metric_attention(in_channels=256, inner_channels=256, num_convs=2)
        self.compare_c3 = metric_attention(in_channels=256, inner_channels=256, num_convs=2)
        self.compare_c4 = metric_attention(in_channels=256, inner_channels=256, num_convs=2)
        self.compare_c5 = metric_attention(in_channels=256, inner_channels=256, num_convs=2)

        # feature fusion
        self.fused = featureFused(inchannels=256, outchannels=256)

        # change head
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),

        )

        # building head
        self.buildingHead = buildingTask(inchannels=256)


    def forward(self, input1, input2):
        feature_x1 = self.backbone(input1)
        feature_x2 = self.backbone(input2)

        building_logit = self.buildingHead(feature_x1)
        building_logit = torch.clamp(building_logit, 1e-6, 1-1e-6)

        compare_out2, metric_out2 = self.compare_c2(feature_x1[0], feature_x2[0])
        compare_out3, metric_out3 = self.compare_c3(feature_x1[1], feature_x2[1])
        compare_out4, metric_out4 = self.compare_c4(feature_x1[2], feature_x2[2])
        compare_out5, metric_out5 = self.compare_c5(feature_x1[3], feature_x2[3])

        compare_out = [compare_out2, compare_out3, compare_out4, compare_out5]
        metric_out = [metric_out2, metric_out3, metric_out4, metric_out5]

        compare_out = self.fused(compare_out)

        compare_out = self.head(compare_out)
        compare_out = torch.clamp(compare_out, 1e-6, 1-1e-6)

        return building_logit, compare_out, metric_out