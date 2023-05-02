import torch
from torch import nn
import torch.nn.functional as F

from .resnet import resnet_encoders
import networks.initialization as init
from networks.modules import BNReLU, ProjectionHead


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
    def __init__(self, encoder_name, num_classes, output_layer_numbers, depth=5, proj_dim=128):
        super().__init__()

        encoder = encoders[encoder_name]['encoder']
        encoder_params = encoders[encoder_name]['params']
        encoder_params.update(depth=depth)
        self.encoder = encoder(**encoder_params)
        self.output_layer_numbers = list(map(int, list(output_layer_numbers)))

        # self.classification_head = WGAP(encoder_params['out_channels'][-1], num_classes)

        in_channels = encoder_params['out_channels'][-1]
        # self.seg_head = SegHead(in_channels, num_classes)
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BNReLU(in_channels, bn_type='torchbn'),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.proj_head = dict()
        if 4 in self.output_layer_numbers:
            in_channels = encoder_params['out_channels'][-1]
            self.proj_head_layer4 = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')
        if 3 in self.output_layer_numbers:
            in_channels = encoder_params['out_channels'][-2]
            self.proj_head_layer3 = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')
        if 2 in self.output_layer_numbers:
            in_channels = encoder_params['out_channels'][-3]
            self.proj_head_layer2 = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')
        if 1 in self.output_layer_numbers:
            in_channels = encoder_params['out_channels'][-4]
            self.proj_head_layer1 = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')

        # self.initialize()

    def initialize(self):
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        features = self.encoder(x)
        # cl_logits = self.classification_head(features[-1])
        seg, out_decoder = self.seg_head(features[-1])

        embed = dict()
        if 4 in self.output_layer_numbers:
            embed['layer4'] = self.proj_head_layer4(features[-1])
            # embed['layer4'] = self.proj_head_layer4(out_decoder)
        if 3 in self.output_layer_numbers:
            embed['layer3'] = self.proj_head_layer3(features[-2])
        if 2 in self.output_layer_numbers:
            embed['layer2'] = self.proj_head_layer2(features[-3])
        if 1 in self.output_layer_numbers:
            embed['layer1'] = self.proj_head_layer1(features[-4])

        return {'seg': seg, 'embed': embed}
