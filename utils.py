import os
import datetime
import logging
from easydict import EasyDict
from torchcam import methods
from tqdm import tqdm
import numpy as np


import torch
import torch.nn.functional as F

from evaluation import MaskEvaluation


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def adjust_learning_rate(args, optimizer, i_iter, max_iterations):
    lr = lr_poly(args.learning_rate, i_iter, max_iterations, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def strip_DataParallel(net):
    if isinstance(net, torch.nn.DataParallel):
        return strip_DataParallel(net.module)
    return net


def save_checkpoint(model, optimizer, opt, epoch, save_file):
    opt.logger.info(f'==> Saving... "{save_file}"')
    opt_dict = {k: v for k, v in opt.__dict__.items() if k not in ["logger", "tb_logger"]}

    state = {
        'opt': opt_dict,
        'model': strip_DataParallel(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def set_up_logger(logs_path, log_file_name=None):
    # logging settings
    logger = logging.getLogger()
    if log_file_name is None:
        log_file_name = str(datetime.datetime.now()).split(".")[0] \
            .replace(" ", "_").replace(":", "_").replace("-", "_") + ".log"
    fileHandler = logging.FileHandler(os.path.join(logs_path, log_file_name), mode="w")
    consoleHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname).1s   %(message)s", datefmt="%H:%M:%S")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.info(f"Created {log_file_name}")
    return logger


def log_parameters(opt, logger):
    logger.info("-" * 10)
    logger.info("Parameters: ")
    opt_dict = opt.__dict__
    longest_key = max(len(k) for k in opt_dict.keys())
    for name, value in opt_dict.items():
        logger.info(f"{name.ljust(longest_key)} = {value}")
    logger.info("-" * 10)


def configure_metadata(metadata_root):
    metadata = EasyDict()
    metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    metadata.image_ids_proxy = os.path.join(metadata_root, 'image_ids_proxy.txt')
    metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
    metadata.localization = os.path.join(metadata_root, 'localization.txt')
    return metadata


def get_cam_extractor(wsol_method, model, target_layer=None, fc_layer=None):
    model.eval()

    if wsol_method.lower() == 'cam':
        return methods.CAM(model, target_layer=target_layer, fc_layer=fc_layer)
    if wsol_method.lower() == 'layercam':
        return methods.LayerCAM(model, target_layer=target_layer)
    if wsol_method.lower() == "gradcam":
        return methods.GradCAM(model, target_layer=target_layer)
    if wsol_method.lower() == "gradcampp":
        return methods.GradCAMpp(model, target_layer=target_layer)
    if wsol_method.lower() == "smoothgradcampp":
        return methods.SmoothGradCAMpp(model, target_layer=target_layer, num_samples=4, std=0.3)
    if wsol_method.lower() == "scorecam":
        return methods.ScoreCAM(model, target_layer=target_layer, batch_size=32)
    if wsol_method.lower() == "xgradcam":
        return methods.XGradCAM(model, target_layer=target_layer)
    if wsol_method.lower() == "iscam":
        return methods.ISCAM(model, target_layer=target_layer)


def is_required_grad(wsol_method):
    if wsol_method.lower() in ['cam', 'scorecam', 'iscam', 'ours']:
        return False
    elif wsol_method.lower() in ['layercam', 'gradcam', 'gradcampp', 'smoothgradcampp', 'xgradcam']:
        return True


def save_pseudo_labels(model, data_loaders, save_path, round, logger, device):
    os.rename(save_path, os.path.join(os.path.dirname(save_path), f'cams_round{round-1}'))
    os.makedirs(save_path, exist_ok=False)

    model = model.eval()
    with torch.no_grad():
        for split in ['train_ps', 'valcl', 'test']:
            evaluator = MaskEvaluation(cam_curve_interval=0.001)
            for data_dict in tqdm(data_loaders[split]):
                image = data_dict['image'].to(device)
                gt_masks = data_dict['mask'].to(device)
                image_name = data_dict['image_id'][0]
                cam_name = f'{os.path.splitext(os.path.basename(image_name))[0]}.npy'

                logits_seg = model(image)['seg']
                np.save(os.path.join(save_path, cam_name),
                        logits_seg[:, 1].squeeze(0).detach().cpu().numpy().astype(float))
                logits_seg = F.interpolate(logits_seg, gt_masks.size()[2:], mode='bicubic', align_corners=False)
                logits_seg = torch.sigmoid(logits_seg[:, 1]).squeeze(0).squeeze(0).detach().cpu().numpy().astype(float)
                # logits_seg = torch.softmax(logits_seg, dim=1)[:, 1].squeeze(0).squeeze(0).cpu().numpy().astype(float)

                mask = gt_masks.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                evaluator.accumulate(logits_seg, mask)
            pxap = evaluator.compute()
            logger.info(f"Round {round} - PXAP on {split if split != 'train_ps' else 'train'} set: {pxap}")

