import logging
import os.path

import torch

from .STDCLModel import STDCLModel
from .unet import UnetNEGEV


def create_model(cfg):
    logger = cfg.logger

    model = None
    if cfg.task.lower() == 'stdcl':
        model = STDCLModel(encoder_name=cfg.encoder_name,
                           num_classes=cfg.num_classes, output_layer_numbers=cfg.output_layer_numbers,
                           proj_dim=cfg.proj_dim, use_aspp=cfg.use_aspp)
    elif cfg.task.lower() == 'unet':
        model = UnetNEGEV(encoder_name=cfg.encoder_name, seg_h_out_channels=2)
    else:
        logger.info(f"{cfg.model_name} model is not valid!")
        raise ValueError(f"{cfg.model_name} model is not valid!")

    if cfg.pretrained is not None:
        logger.info(f"Loading pretrain weights '{cfg.pretrained}'")
        state_dict = torch.load(os.path.join(cfg.pretrained, 'encoder.pt'), map_location=torch.device('cpu'))
        strict = False if cfg.encoder_name in ['inceptionv3', 'vgg16'] else True
        model.encoder.load_state_dict(state_dict, strict=strict)

    return model
