import os
import argparse
from glob import glob

import cv2
import numpy as np
from torchcam.utils import overlay_mask

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from losses.utils import (generate_pseudo_mask_by_cam as generate_pseudo_mask_for_ce,
                          generate_foreground_background_mask as generate_pseudo_mask_for_cl)


def overlay_sampled_pixels_on_image(pseudo_mask, gt_mask_on_image):
    # BGR
    sampled_pixels_ce = np.stack([(pseudo_mask == 0),
                                  np.zeros_like(pseudo_mask),
                                  (pseudo_mask == 1)], axis=-1).astype(np.uint8) * 255
    return cv2.addWeighted(np.asarray(gt_mask_on_image).astype(np.uint8), 0.5,
                           sampled_pixels_ce, 1, 0)


def read_image(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def read_mask(mask_path, mask_size):
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    return cv2.resize(gt_mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)


def read_and_procees_cam(cam_path, cam_size):
    cam_tensor = torch.tensor(np.load(cam_path), dtype=torch.float)
    cam_tensor = F.interpolate(cam_tensor.unsqueeze(0).unsqueeze(0), [cam_size, cam_size],
                               mode='bicubic', align_corners=False)
    return torch.sigmoid(cam_tensor)


def overlay_images(image, mask, alpha=0.4):
    return overlay_mask(to_pil_image(image), to_pil_image(mask), alpha=alpha)[0]


def generate_pseudo_mask(cam_tensor, ignore_index, sample_ratio, generator_function):
    pseudo_mask = generator_function(cam_tensor, ignore_index, sample_ratio)
    return pseudo_mask.squeeze().numpy()


def save_overlay_image(overlay_image, save_path):
    cv2.imwrite(save_path, overlay_image)


def process_image(image_path, resize_size, round, run_dir):
    image_name, extension = os.path.splitext(os.path.basename(image_path))

    image = read_image(image_path, resize_size)

    mask_path = os.path.join(os.path.dirname(image_path), f'{image_name}_anno{extension}')
    gt_mask = read_mask(mask_path, resize_size)

    mask_on_image = overlay_images(image, gt_mask, alpha=0.4)

    cam_path = os.path.join(args.cam_dir, f'cams_round{round}', f'{image_name}.npy')
    cam_tensor = read_and_procees_cam(cam_path, resize_size)

    cl_mask = generate_pseudo_mask(cam_tensor, ignore_index=255, sample_ratio=0.013,
                                   generator_function=generate_pseudo_mask_for_cl)
    ce_mask = generate_pseudo_mask(cam_tensor, ignore_index=255, sample_ratio=0.2,
                                   generator_function=generate_pseudo_mask_for_ce)

    sampled_pixels_on_images_ce = overlay_sampled_pixels_on_image(ce_mask, mask_on_image)
    sampled_pixels_on_images_cl = overlay_sampled_pixels_on_image(cl_mask, mask_on_image)

    save_overlay_image(sampled_pixels_on_images_ce,
                       os.path.join(run_dir, f'{image_name}_round{round}_ce.png'))
    save_overlay_image(sampled_pixels_on_images_cl,
                       os.path.join(run_dir, f'{image_name}_round{round}_cl.png'))


def main(cfg):
    cfg.run_dir = os.path.join(cfg.run_dir, cfg.split)
    os.makedirs(cfg.run_dir, exist_ok=True)
    pattern = f'{cfg.split}*.bmp' if cfg.split != 'train+test' else '*.bmp'
    file_paths = glob(os.path.join(cfg.image_dir, pattern))
    image_paths = [file_name for file_name in file_paths if 'anno' not in file_name]

    for counter, image_path in enumerate(image_paths):
        for round_num in range(1, cfg.num_rounds + 1):
            process_image(image_path, cfg.resize_size, round_num, cfg.run_dir)
        print(f'{counter + 1}/{len(image_paths)}: {os.path.basename(image_path)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_dir', type=str, help='')
    parser.add_argument('--cam_dir', type=str, help='')
    parser.add_argument('--num_rounds', type=int, default=4, help='which round are CAMs related to')
    parser.add_argument('--run_dir', type=str, default='sampled_pixels_on_images', help='')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'train+test'], help='')
    parser.add_argument('--resize_size', type=int, default=224, help='resize size')
    args = parser.parse_args()
    main(args)






