import os
import argparse
from glob import glob
from collections import OrderedDict

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
    return (sample_evaluator.perf_gist['Dice foreground'],
            sample_evaluator.perf_gist['Dice background'],
            sample_evaluator.perf_gist['mdice_best'])


def get_binary_masks(cam, thresholds, image_name):
    image_name, ext = os.path.splitext(image_name)
    return OrderedDict({f'{image_name}_th{idx}_{thresh}': (cam > thresh).astype(np.uint8) * 255
                        for idx, thresh in enumerate(thresholds)})


def get_overlay_mask(image, bi_mask, colormap='jet', alpha=0.6,
                     additional_height=30, font_size=10, text_color='black', text=None):
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


def save_fn_fp_masks(image, gt_mask, best_thresh_mask, save_path):
    save_path = os.path.join(save_path, 'fp_fn_masks')
    os.makedirs(save_path, exist_ok=True)

    mask_name, bi_mask = best_thresh_mask
    if bi_mask.max() == 255:
        best_thresh_mask = (bi_mask / 255).astype(np.uint8)
    # Create false negative mask and false positive mask
    fn_mask = (gt_mask & (1 - best_thresh_mask)) * 255
    fp_mask = ((1 - gt_mask) & best_thresh_mask) * 255
    # Create red mask for false negatives and blue mask for fave positives (BGR)
    fn_red_mask = np.stack([np.zeros_like(fn_mask)] * 2 + [fn_mask], axis=-1)
    fp_blue_mask = np.stack([fp_mask] + [np.zeros_like(fp_mask)] * 2, axis=-1)
    fn_fp_mask = np.stack([fp_mask] + [np.zeros_like(fp_mask)] + [fn_mask], axis=-1)
    # Overlay false negative red mask on the image
    mask_on_image, _ = overlay_mask(to_pil_image(image), to_pil_image(gt_mask.astype(np.float32)), alpha=0.6)
    fn_mask_on_image = cv2.addWeighted(np.asarray(mask_on_image).astype(np.uint8), 0.4, fn_red_mask, 1, 0)
    fp_mask_on_image = cv2.addWeighted(np.asarray(mask_on_image).astype(np.uint8), 0.4, fp_blue_mask, 1, 0)
    fp_fn_mask_on_image = cv2.addWeighted(np.asarray(mask_on_image).astype(np.uint8), 0.4, fn_fp_mask, 1, 0)

    # Save the results
    mask_on_image.save(os.path.join(save_path, f'{mask_name}_gt_mask.png'))
    cv2.imwrite(os.path.join(save_path, f'{mask_name}_fn_mask.png'), fn_mask_on_image)
    cv2.imwrite(os.path.join(save_path, f'{mask_name}_fp_mask.png'), fp_mask_on_image)
    cv2.imwrite(os.path.join(save_path, f'{mask_name}_fp_fn_mask.png'), fp_fn_mask_on_image)


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

        cam = np.load(cam_path)
        if cfg.method != 'negev':
            cam = torch.Tensor(cam).unsqueeze(0).unsqueeze(0)
            cam = F.interpolate(cam,
                                (cfg.resize_size, cfg.resize_size),
                                mode='bicubic',
                                align_corners=False).squeeze()
            cam = torch.sigmoid(cam).numpy().astype(float)

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
        save_fn_fp_masks(image, gt_mask, binary_mask_dict.popitem(last=True), binary_mask_dir)
        print(f'{idx + 1}/{len(cam_paths)}: {image_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cam_dir', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--binary_mask_dir', type=str, default='binary_masks',
                        help='The path where the generated binary masks will be saved.')
    parser.add_argument('--method', type=str, default='negev', choices=['ours', 'negev'])
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
