import os
from glob import glob

import torch

from .STDCLModel import STDCLModel
from .unet import UnetNEGEV


def find_previous_best_model_path(models_path, pre_round):
    pattern = os.path.join(models_path, f'*valcl*rnd_{pre_round}*.pth')
    weights_paths = glob(pattern)

    if not weights_paths:
        return None

    best_weight_path = max(weights_paths, key=lambda x: float(os.path.splitext(x)[0].split('_')[-1]))
    return best_weight_path


def load_weighs(model, weight_path, strict=True):
    checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=strict)


def create_model(cfg):
    logger = cfg.logger

    model = None
    if cfg.task.lower() == 'stdcl':
        model = STDCLModel(encoder_name=cfg.encoder_name,
                           num_classes=cfg.num_classes, proj_dim=cfg.proj_dim, use_aspp=cfg.use_aspp)
    elif cfg.task.lower() == 'unet':
        model = UnetNEGEV(encoder_name=cfg.encoder_name, seg_h_out_channels=cfg.num_classes)
    else:
        logger.info(f"{cfg.model_name} model is not valid!")
        raise ValueError(f"{cfg.model_name} model is not valid!")

    if cfg.pretrained:
        strict = cfg.encoder_name not in ['inceptionv3', 'vgg16']
        if cfg.load_pre_best and cfg.round != 1:
            weight_path = find_previous_best_model_path(cfg.save_folder, cfg.round-1)
            if weight_path:
                load_weighs(model, weight_path, strict)
                logger.info(f"Loading pretrain weights '{weight_path}'")
            else:
                logger.info(f'No previous best model {weight_path} found.')
        else:
            weight_path = os.path.join(cfg.pretrained, 'encoder.pt')
            if os.path.exists(weight_path):
                load_weighs(model.encoder, weight_path, strict)
                logger.info(f"Loading pretrain weights '{weight_path}'")
            else:
                logger.info(f'Pretrained weights not found at {weight_path}.')

    return model
