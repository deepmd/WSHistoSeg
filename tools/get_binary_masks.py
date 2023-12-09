import os
import argparse
from glob import glob
import numpy as np

import cv2
from PIL import Image, ImageDraw, ImageFont
from torchcam.utils import overlay_mask

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image
from evaluation import MaskEvaluation

import warnings
warnings.filterwarnings('ignore')


def get_best_threshold(cam, mask):
    sample_evaluator = MaskEvaluation(cam_curve_interval=0.001)
    sample_evaluator.accumulate(cam, mask)
    sample_evaluator.compute()
    return sample_evaluator.perf_gist['Best tau'][0]


def get_dice_metric(cam, mask, tau=None):
    sample_evaluator = MaskEvaluation(cam_curve_interval=0.001, best_valid_tau=tau)
    sample_evaluator.accumulate(cam, mask)
    sample_evaluator.compute()
    return (float(sample_evaluator.perf_gist['Dice foreground']),
            float(sample_evaluator.perf_gist['Dice background']),
            float(sample_evaluator.perf_gist['mdice_best']))


def get_binary_masks(cam, thresholds, image_name):
    image_name, ext = os.path.splitext(image_name)
    return {f'{image_name}_th_{thresh}': (cam > thresh).astype(np.uint8) * 255
            for thresh in thresholds}


def get_overlay_mask(image, bi_mask, colormap, alpha, additional_height, font_size, text_color, text=None):
    if np.max(bi_mask) == 255:
        bi_mask = (bi_mask / 255).astype(np.float32)
    bi_mask_on_image, _ = overlay_mask(to_pil_image(image),
                                       to_pil_image(bi_mask),
                                       colormap=colormap,
                                       alpha=alpha)
    if text is None:
        return bi_mask_on_image

    width, height = bi_mask_on_image.size
    final_image = Image.new("RGB", (width, height + additional_height), color='white')
    final_image.paste(bi_mask_on_image, (0, additional_height))
    # Add text information to the combined image
    draw = ImageDraw.Draw(final_image)
    # font = ImageFont.load_default()  # You can specify a different font if needed
    font = ImageFont.truetype('fonts/LiberationSerif-Bold.ttf', font_size)
    # Calculate the size of the text
    text_size = draw.textsize(text, font=font)
    # Determine the position to center the text
    text_position = ((width - text_size[0]) // 2, 0)
    draw.text(text_position, text, fill=text_color, font=font)
    return final_image


def save_masks(image, bi_mask_dict, dice_list, save_dir, mask_on_image,
               colormap, alpha, additional_height, font_size, text_color):
    for idx, (bi_mask_name, bi_mask) in enumerate(bi_mask_dict.items()):
        cv2.imwrite(os.path.join(save_dir, f'{bi_mask_name}.png'), bi_mask)

        if mask_on_image:
            fg_dice, bg_dice, mdice = dice_list[idx]
            text = (f"mDSC: {mdice:.2f} \n"
                    f"(DSC Foreground: {fg_dice:.2f}, DSC Background: {bg_dice:.2f})")
            bi_mask_on_image = get_overlay_mask(image, bi_mask, colormap, alpha,
                                                additional_height, font_size, text_color, text)
            bi_mask_on_image.save(os.path.join(save_dir, f'{bi_mask_name}_ovly.png'))


def main(cfg):
    if not os.path.exists(cfg.cam_dir):
        raise ValueError(f'{cfg.cam_dir} is not exist.')

    binary_mask_dir = os.path.join(cfg.binary_mask_dir, cfg.method, cfg.split)
    os.makedirs(binary_mask_dir, exist_ok=True)

    pattern = f'{cfg.split}*.npy' if cfg.split != 'train+test' else '*.npy'
    cam_paths = glob(os.path.join(cfg.cam_dir, pattern))

    for idx, cam_path in enumerate(cam_paths):
        image_name = f'{os.path.splitext(os.path.basename(cam_path))[0]}.bmp'
        image_path = os.path.join(cfg.image_dir, image_name)
        mask_name = f'{os.path.splitext(os.path.basename(cam_path))[0]}_anno.bmp'
        mask_path = os.path.join(cfg.image_dir, mask_name)
        thresh_list = list(map(float, cfg.thresholds.split(','))) if cfg.thresholds != '' else []

        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (cfg.resize_size, cfg.resize_size), interpolation=cv2.INTER_LINEAR)
        # height, width, _ = image.shape

        cam = torch.Tensor(np.load(cam_path))
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                            (cfg.resize_size, cfg.resize_size),
                            mode='bicubic',
                            align_corners=False)
        cam = torch.sigmoid(cam.squeeze()).detach().cpu().numpy().astype(float)

        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0.5
        gt_mask = cv2.resize(gt_mask.astype(np.uint8),
                             (cfg.resize_size, cfg.resize_size),
                             interpolation=cv2.INTER_NEAREST)
        thresh_list.append(get_best_threshold(cam, gt_mask))

        binary_mask_dict = get_binary_masks(cam, thresh_list, image_name)

        dice_list = [get_dice_metric(cam, gt_mask, thresh) for thresh in thresh_list]
        save_masks(image, binary_mask_dict, dice_list, binary_mask_dir,
                   cfg.binary_mask_on_image, cfg.colormap, cfg.alpha,
                   cfg.additional_height, cfg.font_size, cfg.text_color)
        print(f'{idx + 1}/{len(cam_paths)}: {image_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cam_dir', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--binary_mask_dir', type=str, default='binary_masks1',
                        help='The path where the generated binary masks will be saved.')
    parser.add_argument('--method', type=str, default='ours', choices=['ours', 'gradcam', 'negev'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test', 'train+test'])
    parser.add_argument('--thresholds', type=str, default='')
    parser.add_argument('--binary_mask_on_image', action='store_true',
                        help='Whether to overlay the binary masks on the images or not.')
    parser.add_argument('--colormap', type=str, default='jet',
                        help='Specify the colormap to use for drawing text on images.')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--additional_height', type=int, default=30)
    parser.add_argument('--font_size', type=int, default=10,
                        help='Specify the font_size to use for drawing text on images.')
    parser.add_argument('--text_color', type=str, default='black',
                        help='Specify the text_color to use for drawing text on images.')
    parser.add_argument('--resize_size', type=int, default=224, help='resize size')
    args = parser.parse_args()
    main(args)
