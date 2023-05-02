import torch
from torch import nn
import torch.nn.functional as F
import networks.modules as md
from .modules import BNReLU, ProjectionHead
from networks.resnet import resnet_encoders
import networks.initialization as init


encoders = {}
encoders.update(resnet_encoders)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type,
                                       in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    input=x,
                    size=skip.shape[2:],
                    mode='bilinear',
                    align_corners=True
                )
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetNEGEVDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False

    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for "
                "{} blocks.".format(n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(
                in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward_negev(self, *features, cam):
        raise NotImplementedError

        # remove first skip with same spatial resolution
        features_ = features[1:]
        # reverse channels to start from head of encoder
        features_ = features_[::-1]

        head = features_[0]
        skips = features_[1:]

        x = self.center(head * cam)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

    def forward_reconstruction(self, *features):
        # remove first skip with same spatial resolution
        features_ = features[1:]
        # reverse channels to start from head of encoder
        features_ = features_[::-1]

        head = features_[0]
        skips = features_[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

    def forward_seg_old(self, *features):
        raise NotImplementedError

        # remove first skip with same spatial resolution
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x

    def forward(self, *features):
        # remove first skip with same spatial resolution
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class SegmentationHead(nn.Sequential):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 activation=None,
                 upsampling=1
                 ):
        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2
                           )
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = md.Activation(activation)
        super().__init__(conv2d, upsampling, activation)


def count_params(model: torch.nn.Module):
    return sum([p.numel() for p in model.parameters()])


class FCAMModel(torch.nn.Module):
    def initialize(self):
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

        if self.decoder is not None:
            init.initialize_decoder(self.decoder)

        if self.segmentation_head is not None:
            init.initialize_head(self.segmentation_head)

        if self.reconstruction_head is not None:
            init.initialize_head(self.reconstruction_head)

    def get_reconstructed_img(self, *features):
        assert self.reconstruction_head
        return self.reconstruction_head(self.decoder.forward_reconstruction(
            *features))

    def forward(self, x):
        x_shape = x.shape

        if self.scale_in != 1.:
            raise ValueError

            # h, w = x_shape[2], x_shape[3]
            # x = F.interpolate(
            #     input=x,
            #     size=[int(h * self.scale_in), int(w * self.scale_in)],
            #     mode='bilinear',
            #     align_corners=True
            # )

        self.x_in = x

        features = self.encoder(x)
        if self.freeze_cl:
            features = [f.detach() for f in features]

        # cl_logits = self.classification_head(features[-1])
        decoder_output = self.decoder(*features)
        seg = self.segmentation_head(decoder_output)
        embed = self.proj_head(decoder_output)

        if seg.shape[2:] != x_shape[2:]:
            seg = F.interpolate(
                input=seg,
                size=x_shape[2:],
                mode='bilinear',
                align_corners=True
            )

        if embed.shape[2:] != x_shape[2:]:
            embed = F.interpolate(
                input=embed,
                size=x_shape[2:],
                mode='bilinear',
                align_corners=True
            )

        # im_recon = None
        # if self.im_rec:
        #     im_recon = self.get_reconstructed_img(*features)

        # self.cams = fcams.detach()

        return {'seg': seg, 'embed': embed}

    def train(self, mode=True):
        super(FCAMModel, self).train(mode=mode)

        if self.freeze_cl:
            self.freeze_classifier()

        return self

    def freeze_classifier(self):
        assert self.freeze_cl

        for module in (self.encoder.modules()):

            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

        for module in (self.classification_head.modules()):
            for param in module.parameters():
                param.requires_grad = False

            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

            if isinstance(module, torch.nn.Dropout):
                module.eval()

    def assert_cl_is_frozen(self):
        assert self.freeze_cl

        for module in (self.encoder.modules()):
            for param in module.parameters():
                assert not param.requires_grad

            if isinstance(module, torch.nn.BatchNorm2d):
                assert not module.training

            if isinstance(module, torch.nn.Dropout):
                assert not module.training

        for module in (self.classification_head.modules()):
            for param in module.parameters():
                assert not param.requires_grad

            if isinstance(module, torch.nn.BatchNorm2d):
                assert not module.training

            if isinstance(module, torch.nn.Dropout):
                assert not module.training

        return True

    def __str__(self):
        return "{}. Task: {}. Supp.BACK: {}. Freeze CL: {}. " \
               "IMG-RECON: {}:".format(
                self.name, self.task,
                self.classification_head.support_background,
                self.freeze_cl, self.im_rec
                )

    def get_info_nbr_params(self) -> str:
        info = self.__str__() + ' \n NBR-PARAMS: \n'

        if self.encoder:
            info += '\tEncoder [{}]: {}. \n'.format(
                self.encoder.name,  count_params(self.encoder))

        if self.classification_head:
            info += '\tClassification head [{}]: {}. \n'.format(
                self.classification_head.name,
                count_params(self.classification_head))

        if self.decoder:
            info += '\tDecoder: {}. \n'.format(
                count_params(self.decoder))

        if self.segmentation_head:
            info += '\tSegmentation head: {}. \n'.format(
                count_params(self.classification_head))

        if self.reconstruction_head:
            info += '\tReconstruction head: {}. \n'.format(
                count_params(self.reconstruction_head))

        info += '\tTotal: {}. \n'.format(count_params(self))

        return info


class NEGEVModel(FCAMModel):
    pass


class UnetNEGEV(NEGEVModel):
    """
    NEGEV using U-Net like.
    Unet_ is a fully convolution neural network for image semantic
    segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract
    features of different spatial
    resolution (skip connections) which are used by decoder to define accurate
    segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an
            encoder (a.k.a backbone) to extract features of different spatial
            resolution
        encoder_depth: A number of stages used in encoder in range [3, 5].
        Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for
            depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W),
            (N, C, H // 2, W // 2)] and so on). Default is 5
        encoder_weights: One of **None** (random initialization),
        **"imagenet"** (pre-training on ImageNet) and other pretrained
        weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels**
            parameter for convolutions used in decoder. Length of the list
            should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D
            and Activation layers is used. If **"inplace"** InplaceABN will
            be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model.
            Available options are **None** and **scse**.
            SCSE paper - https://arxiv.org/abs/1808.08127
        in_channels: A number of input channels for the model, default is 3
            (RGB images)
        classes: A number of classes for output mask (or you can think as a
            number of channels of output mask). Useful ONLY for the task
            constants.SEG.
        activation: An activation function to apply after the final convolution
            layer.
            Available options are **"sigmoid"**, **"softmax"**,
            **"logsoftmax"**, **"tanh"**, **"identity"**, **callable** and
            **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output
            (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default).
            Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply
                "sigmoid"/"softmax" (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels=(256, 128, 64, 32, 16),
        decoder_attention_type=None,
        in_channels: int = 3,
        seg_h_out_channels: int = 1,
        activation=None,
        aux_params=None,
        scale_in: float = 1.,
        freeze_cl: bool = False,
        im_rec: bool = False,
        img_range="tanh",
        proj_dim=128,
    ):
        super().__init__()

        self.freeze_cl = freeze_cl
        assert scale_in > 0.
        self.scale_in = float(scale_in)

        self.im_rec = im_rec
        self.img_range = img_range

        self.x_in = None

        encoder = encoders[encoder_name]['encoder']
        encoder_params = encoders[encoder_name]['params']
        encoder_params.update(depth=encoder_depth)
        self.encoder = encoder(**encoder_params)

        self.decoder = UnetNEGEVDecoder(
            encoder_channels=encoder_params['out_channels'],
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type
        )

        self.classification_head = None
        # assert aux_params is not None, 'ERROR'
        # pooling_head = aux_params['pooling_head']
        # aux_params.pop('pooling_head')

        # self.classification_head = poolings.__dict__[pooling_head](
        #     in_channels=self.encoder.out_channels[-1], **aux_params
        # )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=seg_h_out_channels,
            activation=activation,
            kernel_size=3,
        )

        self.reconstruction_head = None
        # if self.im_rec:
        #     self.reconstruction_head = ReconstructionHead(
        #         in_channels=decoder_channels[-1],
        #         out_channels=in_channels,
        #         activation=self.img_range,
        #         kernel_size=3,
        #     )

        # self.cams = None

        in_channels = decoder_channels[-1]
        self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')

        self.name = "u-{}".format(encoder_name)
        self.initialize()

