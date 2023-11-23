import os
import argparse
from glob import glob
import numpy as np
import cv2

from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_dir', type=str, help='')
    parser.add_argument('--cam_dir', type=str, help='')
    parser.add_argument('--run_dir', type=str, help='')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'train+test'], help='')
    parser.add_argument('--mask_on_image', action='store_true',
                        help='Specify whether to overlay ground-truth masks on images or not.')
    cfg = parser.parse_args()

    cfg.run_dir = os.path.join(cfg.run_dir, cfg.split)
    os.makedirs(cfg.run_dir, exist_ok=True)
    pattern = f'{cfg.split}*.bmp' if cfg.split != 'train+test' else '*.bmp'
    file_paths = glob(os.path.join(cfg.image_dir, pattern))
    image_paths = [file_name for file_name in file_paths if 'anno' not in file_name]
    mask_paths = [image_path.replace('.', '_anno.') for image_path in image_paths]
    cam_names = [os.path.basename(image_path).replace('.bmp', '.npy') for image_path in image_paths]
    cam_paths = [os.path.join(cfg.cam_dir, cam_name) for cam_name in cam_names]

    for counter, (image_path, mask_path, cam_path) in enumerate(zip(image_paths, mask_paths, cam_paths)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        image_name, extension = os.path.splitext(os.path.basename(image_path))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0.5).astype(np.float32)

        if cfg.mask_on_image:
            mask_on_image, _ = overlay_mask(to_pil_image(image), to_pil_image(mask), alpha=0.6)
            out_path = os.path.join(cfg.run_dir, f'{image_name}_mask.png')
            mask_on_image.save(out_path)

        cam = np.load(cam_path).astype(np.float32)
        cam_on_image, _ = overlay_mask(to_pil_image(image), to_pil_image(cam), alpha=0.6)
        out_path = os.path.join(cfg.run_dir, f'{image_name}_cam.png')
        cam_on_image.save(out_path)
        print(f'{counter+1}/{len(image_paths)}')



