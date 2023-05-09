import os
from PIL import Image
import numpy as np
import cv2
from easydict import EasyDict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class WsolDataset(Dataset):
    def __init__(self, data_root, metadata_root, transforms=None,
                 use_pseudo_masks=False, output_layer_numbers="34",
                 fore_threshold=-1, back_threshold=-1, ignore_index=-1):
        super(WsolDataset, self).__init__()
        self.data_root = data_root
        self.use_pseudo_masks = use_pseudo_masks
        self.metadata = WsolDataset.configure_metadata(metadata_root)
        self.image_ids = WsolDataset.get_image_ids(self.metadata)
        self.image_labels = WsolDataset.get_class_labels(self.metadata)
        # self.num_sample_per_class = num_sample_per_class
        self.index_id = {image_id: idx for idx, image_id in enumerate(self.image_ids)}
        self.mask_paths = WsolDataset.get_mask_paths(self.metadata)[0] if not self.use_pseudo_masks else None
        self.cam_paths = WsolDataset.get_cam_paths(self.metadata)
        self.transforms = transforms
        self.output_layer_numbers = output_layer_numbers
        self.fore_threshold = fore_threshold
        self.back_threshold = back_threshold
        self.ignore_index = ignore_index

    @staticmethod
    def configure_metadata(metadata_root):
        metadata = EasyDict()
        metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
        metadata.image_ids_proxy = os.path.join(metadata_root, 'image_ids_proxy.txt')
        metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
        metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
        metadata.localization = os.path.join(metadata_root, 'localization.txt')
        metadata.image_cams = os.path.join(metadata_root, 'image_cams.txt')
        return metadata

    @staticmethod
    def get_image_ids(metadata, proxy=False):
        """
        image_ids.txt has the structure

        <path>
        path/to/image1.jpg
        path/to/image2.jpg
        path/to/image3.jpg
        ...
        """
        image_ids = []
        suffix = '_proxy' if proxy else ''
        with open(metadata['image_ids' + suffix]) as f:
            for line in f.readlines():
                image_ids.append(line.strip('\n'))
        return image_ids

    @staticmethod
    def get_class_labels(metadata):
        """
        class_labels.txt has the structure

        <path>,<integer_class_label>
        path/to/image1.jpg,0
        path/to/image2.jpg,1
        path/to/image3.jpg,1
        ...
        """
        class_labels = {}
        with open(metadata.class_labels) as f:
            for line in f.readlines():
                image_id, class_label_string = line.strip('\n').split(',')
                class_labels[image_id] = int(class_label_string)
        return class_labels

    @staticmethod
    def get_mask_paths(metadata, use_pseudo_masks=False):
        """
        localization.txt (for masks) has the structure

        <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
        path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
        path/to/image1.jpg,path/to/mask1b.png,
        path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
        path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
        ...

        One image may contain multiple masks (multiple mask paths for same image).
        One image contains only one ignore mask.
        """
        mask_paths = {}
        ignore_paths = {}
        with open(metadata.localization) as f:

            for line in f.readlines():
                image_id, mask_path, ignore_path = line.strip('\n').split(',')
                if image_id in mask_paths:
                    mask_paths[image_id].append(mask_path)
                    assert (len(ignore_path) == 0)
                else:
                    mask_paths[image_id] = [mask_path]
                    ignore_paths[image_id] = ignore_path
        return mask_paths, ignore_paths

    @staticmethod
    def get_cam_paths(metadata):
        cam_paths = {}
        with open(metadata.image_cams) as f:
            for line in f.readlines():
                line_parts = line.strip('\n').split(',')
                image_id, cam_file_paths = line_parts[0], line_parts[1:]
                cam_paths[image_id] = cam_file_paths
        return cam_paths

    def __len__(self):
        return len(self.image_ids)

    def get_cam(self, image_id, image_size, layer_numbers):
        cams = {'layer1': None, 'layer2': None, 'layer3': None, 'layer4': None}
        layer_numbers = map(int, list(layer_numbers))
        for layer_number in layer_numbers:
            cam_file = os.path.join(self.data_root, self.cam_paths[image_id][layer_number-1])
            cam = np.load(cam_file)
            # mask = (cam > 0.5).astype(np.uint8)
            cam = torch.from_numpy(cam).float()
            # mask = mask.unsqueeze(0).unsqueeze(0)
            cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), image_size, mode='bicubic', align_corners=True)
            cams[f'layer{layer_number}'] = cam.squeeze(0)  # 1, H, W
        return cams

    def get_mask(self, image_id):
        mask_file = os.path.join(self.data_root, self.mask_paths[image_id][0])
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE).astype(np.float)
        mask = (mask > 0.5).astype(np.uint8)
        mask = torch.from_numpy(mask).long()
        mask = mask.unsqueeze(0)  # 1, H, W
        return mask

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = Image.open(os.path.join(self.data_root, image_id))
        image = image.convert('RGB')  # H, W, 3
        raw_image = image.copy()

        cams = self.get_cam(image_id, image.size, layer_numbers=self.output_layer_numbers)
        masks = {'layer1': None, 'layer2': None, 'layer3': None, 'layer4': None}
        masks['layer4'] = self.get_mask(image_id) if not self.use_pseudo_masks else None
        # mask = (pseudo_mask > 0.5).astype(np.uint8)
        # mask = np.expand_dims(mask, axis=0)  # 1, H, W
        # mask = Image.fromarray(mask)
        # mask = mask.resize(image.size, resample=Image.NEAREST)

        if self.transforms:
            image, raw_image, \
            cams['layer4'], cams['layer3'], cams['layer2'], cams['layer1'], \
            masks['layer4'], masks['layer3'], masks['layer2'], masks['layer1'] = \
                self.transforms(image, raw_image, cams['layer4'], cams['layer3'], cams['layer2'], cams['layer1'],
                                masks['layer4'], masks['layer3'], masks['layer2'], masks['layer1'])

        if self.use_pseudo_masks:
            for layer_name, cam in cams.items():
                if cam is not None:
                    # cam = (cam - cam.min())/(cam.max() - cam.min())
                    reshaped_cam = cam.reshape(cam.size()[0], -1)
                    sorted_cam = torch.sort(reshaped_cam, dim=1)[0]
                    x = int(0.2 * sorted_cam.size()[1])
                    self.fore_threshold = sorted_cam[0, -x]
                    self.back_threshold = sorted_cam[0, x]

                    mask = torch.ones_like(cam) * self.ignore_index
                    if self.fore_threshold != -1:
                        mask[cam > self.fore_threshold] = 1
                    if self.back_threshold != -1:
                        mask[cam <= self.back_threshold] = 0
                    # mask = F.interpolate(mask.unsqueeze(0), image.size()[1:], mode='nearest')
                    masks[layer_name] = mask.long()

            # mask = np.array(mask)
            # cam = mask
            # final_mask = np.ones_like(mask) * self.ignore_index
            # if self.fore_threshold != -1:
            #     final_mask[mask > self.fore_threshold] = 1
            # if self.back_threshold != -1:
            #     final_mask[mask <= self.back_threshold] = 0
            # mask = torch.from_numpy(final_mask).long()
            # mask = mask.unsqueeze(0)  # 1, H, W

        raw_image = np.array(raw_image, dtype=np.float32)  # h, w, 3
        raw_image = torch.from_numpy(raw_image).permute(2, 0, 1)  # 3, h, w

        masks = {key: value for key, value in masks.items() if value is not None}
        cams = {key: value for key, value in cams.items() if value is not None}

        return {
            'image': image,
            'label': image_label,
            'image_id': image_id,
            'raw_image': raw_image,
            'mask': masks,
            'cam': cams,
        }








