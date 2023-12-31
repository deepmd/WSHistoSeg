import torch
from torch import nn
import torch.nn.functional as F

from .resnet import resnet_encoders
import networks.initialization as init
from networks.modules import BNReLU, ProjectionHead
from networks.aspp import ASPP


encoders = {}
encoders.update(resnet_encoders)


class WGAP(nn.Module):
    """ https://arxiv.org/pdf/1512.04150.pdf """
    def __init__(self, in_channels, classes):
        super(WGAP, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, classes)

    @property
    def builtin_cam(self):
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        return logits


# class SegHead(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(SegHead, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
#         self.bn_relu = BNReLU(in_channels, bn_type='torchbn')
#         self.dropout = nn.Dropout2d(0.10)
#         self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         out = x = self.bn_relu(x)
#         x = self.dropout(x)
#         x = self.conv2(x)
#         return x, out


class STDCLModel(nn.Module):
    def __init__(self, encoder_name, num_classes, depth=5, proj_dim=128, use_aspp=False):
        super().__init__()

        encoder = encoders[encoder_name]['encoder']
        encoder_params = encoders[encoder_name]['params']
        encoder_params.update(depth=depth)
        self.encoder = encoder(**encoder_params)

        # self.classification_head = WGAP(encoder_params['out_channels'][-1], num_classes)

        self.use_aspp = use_aspp
        in_channels = encoder_params['out_channels'][-1]
        if self.use_aspp:
            self.aspp = ASPP(encoder_name, output_stride=8, BatchNorm=nn.BatchNorm2d)
            self.seg_head = nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=False),
                BNReLU(256, bn_type='torchbn'),
                nn.Dropout2d(0.10),
                nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.proj_head = ProjectionHead(dim_in=1280, proj_dim=proj_dim, proj='linear')
        else:
            self.seg_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                BNReLU(in_channels, bn_type='torchbn'),
                nn.Dropout2d(0.10),
                nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
            # self.seg_head = SegHead(in_channels, num_classes)
            self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')

        # self.initialize()

    def initialize(self):
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        features = self.encoder(x)
        # cl_logits = self.classification_head(features[-1])
        x = features[-1]
        if self.use_aspp:
            x = self.aspp(x)
        seg = self.seg_head(x)
        embed = self.proj_head(x)
        return {'seg': seg, 'embed': embed}
