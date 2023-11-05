import os
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .get_paths import *


class WsolDataset(Dataset):
    def __init__(self, data_root, metadata_root, suffix=None, transforms=None):
        super(WsolDataset, self).__init__()
        self.data_root = data_root
        self.suffix = suffix
        self.image_ids = get_image_ids(metadata_root, suffix)
        self.image_labels = get_class_labels(metadata_root)
        self.mask_paths = get_mask_paths(metadata_root)[0]
        self.cam_paths = get_cam_paths(metadata_root)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def get_cam(self, image_id, image_size):
        cam = None
        cam_path = os.path.join(self.data_root, self.cam_paths[image_id][3])
        if os.path.isfile(cam_path):
            cam = np.load(cam_path)
            cam = torch.from_numpy(cam).float().unsqueeze(0)
            # cam = F.interpolate(cam.unsqueeze(0), image_size, mode='bicubic', align_corners=True)
            # cam = cam.squeeze(0)  # 1, H, W
        return cam

    def get_mask(self, image_id):
        mask_path = os.path.join(self.data_root, self.mask_paths[image_id][0])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float)
        mask = (mask > 0.5).astype(np.uint8)
        mask = torch.from_numpy(mask).long()
        mask = mask.unsqueeze(0)  # 1, H, W
        return mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # read image
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')  # H, W, 3

        # a copy of image without gitter color
        raw_image = image.copy()

        # image label
        image_label = self.image_labels[image_id]

        # read mask
        mask = self.get_mask(image_id)

        # read cam
        cam = self.get_cam(image_id, image.size)

        if self.transforms:
                image, raw_image, cam, mask = self.transforms(image, raw_image, cam, mask)

        # raw_image = np.array(raw_image, dtype=np.float32)  # h, w, 3
        # raw_image = torch.from_numpy(raw_image).permute(2, 0, 1)  # 3, h, w

        if cam is not None:
            return {
                'image': image,
                'label': image_label,
                'image_id': image_id,
                'mask': mask,
                'cam': cam,
                'suffix': self.suffix
            }
        return {
            'image': image,
            'label': image_label,
            'image_id': image_id,
            'mask': mask,
            'suffix': self.suffix
        }










