import sys
import os
from glob import glob
import numpy as np
import cv2

import torch


def get_pseudo_binary_mask(x, w, sigma):
    """
    Compute a mask by applying a sigmoid function.
    The mask is not binary but pseudo-binary (values are close to 0/1).

    :param x: tensor of size (batch_size, 1, h, w), cont    ain the feature
     map representing the mask.
    :return: tensor, mask. with size (nbr_batch, 1, h, w).
    """
    # wrong: x.min() .max() operates over the entire tensor.
    # it should be done over each sample.
    x = (x - x.min()) / (x.max() - x.min())
    return torch.sigmoid(w * (x - sigma))


if __name__=='__main__':
    cams_dir = sys.argv[1]
    masks_dir = sys.argv[2]
    w = 5.0
    sigma = 0.15

    os.makedirs(masks_dir, exist_ok=True)

    cam_files = glob(os.path.join(cams_dir, "*.npy"))
    for cam_num, cam_file in enumerate(cam_files, start=1):
        cam = np.load(cam_file)
        mask = np.ones_like(cam)
        mask[cam <= 0.5] = 0
        # cam = (cam - cam.min()) / (cam.max() - cam.min())
        # mask = np.ones_like(cam) * 255
        # mask[cam > 0.5] = 1
        # mask[cam < 0.2] = 0
        # mask = get_pseudo_binary_mask(torch.from_numpy(cam), w, sigma).numpy()
        # mask = Image.fromarray(mask, "L")

        mask_name = os.path.basename(cam_file).replace('.npy', '_mask.png')
        mask_path = os.path.join(masks_dir, mask_name)
        cv2.imwrite(mask_path, mask * 255)

        print(f"{cam_num}/{len(cam_files)}")

