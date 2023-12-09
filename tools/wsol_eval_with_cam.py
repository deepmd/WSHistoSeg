import os
import argparse
from glob import glob

import numpy as np
import cv2

import torch
from torch.nn import functional as F
from utils import set_up_logger
from evaluation import MaskEvaluation
np.seterr('ignore')


def main(cfg):
    out_path = os.path.join(cfg.run_dir, cfg.method)
    os.makedirs(out_path, exist_ok=True)
    logger = set_up_logger(out_path, log_file_name=f'sample_metrics_{cfg.split}')

    split_evaluator = MaskEvaluation(cam_curve_interval=0.001)
    pattern = f'{cfg.split}*.npy' if cfg.split != 'train+test' else '*.npy'
    cam_paths = glob(os.path.join(cfg.cam_dir, pattern))
    for idx, cam_path in enumerate(cam_paths):
        sample_evaluator = MaskEvaluation(cam_curve_interval=0.001)

        image_name = os.path.splitext(os.path.basename(cam_path))[0]
        mask_path = os.path.join(cfg.mask_dir, f'{image_name}_anno.bmp')

        cam = np.load(cam_path)
        cam = torch.Tensor(cam)
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            (cfg.resize_size, cfg.resize_size),
                            mode='bicubic',
                            align_corners=False)
        cam = torch.sigmoid(cam.squeeze()).detach().cpu().numpy().astype(float)

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0.5
        gt_mask = torch.Tensor(gt_mask)
        gt_mask = F.interpolate(gt_mask.unsqueeze(0).unsqueeze(0),
                                (cfg.resize_size, cfg.resize_size),
                                mode='nearest')
        gt_mask = gt_mask.squeeze().detach().cpu().numpy().astype(float)

        split_evaluator.accumulate(cam, gt_mask)
        sample_evaluator.accumulate(cam, gt_mask)
        sample_evaluator.compute()
        sample_metrics = sample_evaluator.perf_gist
        log_messages = [f'\n{idx}/{len(cam_paths)}: {image_name}']
        log_messages.extend(
            f'{key}: {value}'
            for key, value in sample_metrics.items()
            if not isinstance(value, np.ndarray)
        )
        logger.info('\n'.join(log_messages))

    split_evaluator.compute()
    split_metrics = split_evaluator.perf_gist

    np.save(os.path.join(cfg.run_dir, cfg.method, f'split_metrics_{cfg.split}_{cfg.method}.npy'), split_metrics)

    save_path = os.path.join(cfg.run_dir, cfg.method, f'split_metrics_{cfg.split}.txt')
    log_messages = [f'{key}: {value}' for key, value in split_metrics.items() if not isinstance(value, np.ndarray)]
    with open(save_path, 'w') as file:
        file.write('\n'.join(log_messages))
        # for key, value in split_metrics.items():
        #     if not isinstance(value, np.ndarray):
        #         f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cam_dir', type=str, help='')
    parser.add_argument('--mask_dir', type=str, help='')
    parser.add_argument('--run_dir', type=str, default="cams_evals", help='')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'train+test'], help='')
    parser.add_argument('--method', type=str, default='ours', help='')
    parser.add_argument('--resize_size', type=int, default=224, help='resize size')
    args = parser.parse_args()
    main(args)
