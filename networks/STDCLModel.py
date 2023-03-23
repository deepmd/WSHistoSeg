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


class STDCLModel(nn.Module):
    def __init__(self, encoder_name, num_classes, depth=5, proj_dim=128):
        super().__init__()

        encoder = encoders[encoder_name]['encoder']
        encoder_params = encoders[encoder_name]['params']
        encoder_params.update(depth=depth)
        self.encoder = encoder(**encoder_params)

        # self.classification_head = WGAP(encoder_params['out_channels'][-1], num_classes)

        in_channels = encoder_params['out_channels'][-1]
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            BNReLU(in_channels, bn_type='torchbn'),
            nn.Dropout2d(0.10),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')

        # self.initialize()

    def initialize(self):
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        features = self.encoder(x)
        # cl_logits = self.classification_head(features[-1])
        seg = self.seg_head(features[-1])
        embed = self.proj_head(features[-1])
        return {'seg': seg, 'embed': embed}
