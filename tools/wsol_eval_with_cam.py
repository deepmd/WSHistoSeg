import os
import argparse
from glob import glob

import numpy as np
import cv2

from utils import set_up_logger
from evaluation import MaskEvaluation
np.seterr('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cam_dir', type=str, help='')
    parser.add_argument('--mask_dir', type=str, help='')
    parser.add_argument('--run_dir', type=str, default="cams_evals", help='')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'train+test'], help='')
    parser.add_argument('--method', type=str, default='gradcam', help='')
    parser.add_argument('--resize_size', type=int, default=224, help='resize size')
    cfg = parser.parse_args()

    out_path = os.path.join(cfg.run_dir, cfg.method)
    os.makedirs(out_path, exist_ok=True)
    logger = set_up_logger(out_path, log_file_name=f'sample_metrics_{cfg.split}')

    split_evaluator = MaskEvaluation(cam_curve_interval=0.001)
    pattern = f'{cfg.split}*.npy' if cfg.split != 'train+test' else '*.npy'
    cam_paths = glob(os.path.join(cfg.cam_dir, pattern))
    for idx, cam_path in enumerate(cam_paths):
        sample_evaluator = MaskEvaluation(cam_curve_interval=0.001)

        image_name, extension = os.path.splitext(os.path.basename(cam_path))
        mask_path = os.path.join(cfg.mask_dir, f'{image_name}_anno.bmp')

        cam = np.load(cam_path)
        cam = cv2.resize(cam, (cfg.resize_size, cfg.resize_size), interpolation=cv2.INTER_LINEAR)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0.5).astype(np.uint8)
        mask = cv2.resize(mask, (cfg.resize_size, cfg.resize_size), interpolation=cv2.INTER_NEAREST)

        split_evaluator.accumulate(cam, mask)
        sample_evaluator.accumulate(cam, mask)
        sample_evaluator.compute()
        sample_metrics = sample_evaluator.perf_gist
        log_messages = [f'\n{image_name}']
        for key, value in sample_metrics.items():
            if not isinstance(value, np.ndarray):
                log_messages.append(f'{key}: {value}')
        logger.info('\n'.join(log_messages))

    split_evaluator.compute()
    split_metrics = split_evaluator.perf_gist

    np.save(os.path.join(cfg.run_dir, cfg.method, f'split_metrics_{cfg.split}_{cfg.method}.npy'), split_metrics)
    with open(os.path.join(cfg.run_dir, cfg.method, f'split_metrics_{cfg.split}.txt'), 'w') as f:
        for key, value in split_metrics.items():
            if not isinstance(value, np.ndarray):
                f.write(f"{key}: {value}\n")






