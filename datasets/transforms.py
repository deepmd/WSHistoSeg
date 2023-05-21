import random
from typing import Tuple
import numbers
from collections.abc import Sequence


import torch
from torch import Tensor
from torchvision import transforms
import torchvision.transforms.functional as TF

PROB_THRESHOLD = 0.5  # probability threshold.


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Compose(object):
    def __init__(self, mytransforms: list):
        self.transforms = mytransforms

        for t in mytransforms:
            assert any([isinstance(t, Resize),
                        isinstance(t, RandomCrop),
                        isinstance(t, RandomHorizontalFlip),
                        isinstance(t, RandomVerticalFlip),
                        isinstance(t, transforms.ToTensor),
                        isinstance(t, transforms.Normalize),
                        isinstance(t, transforms.ColorJitter)
                        ]
                       )

    def chec_if_random(self, transf):
        if isinstance(transf, RandomCrop):
            return True

    def __call__(self, img, raw_img, cam, mask):
        for t in self.transforms:
            if isinstance(t, (RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, Resize)):
                img, raw_img, cam, mask = t(img, raw_img, cam, mask)
            else:
                img = t(img)

        return img, raw_img, cam, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class _BasicTransform(object):
    def __call__(self, img, raw_img, cam, mask):
        raise NotImplementedError


class RandomHorizontalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, cam, mask):
        if random.random() < self.p:

            cam_ = cam
            if cam_ is not None:
                cam_ = TF.hflip(cam_)

            mask_ = mask
            if mask_ is not None:
                mask_ = TF.hflip(mask_)

            return TF.hflip(img), TF.hflip(raw_img), cam_, mask_

        return img, raw_img, cam, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, cam, mask):
        if random.random() < self.p:
            cam_ = cam
            if cam_ is not None:
                cam_ = TF.vflip(cam_)

            mask_ = mask
            if mask_ is not None:
                mask_ = TF.vflip(mask_)

            return TF.vflip(img), TF.vflip(raw_img), cam_, mask_

        return img, raw_img, cam, mask

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomCrop(_BasicTransform):
    @staticmethod
    def get_params(img: Tensor, output_size: Tuple[int, int]
                   ) -> Tuple[int, int, int, int]:

        w, h = TF.get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image "
                "size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0,
                 padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two "
                            "dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = TF.get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = TF.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = TF.pad(img, padding, self.fill, self.padding_mode)

        return img

    def __call__(self, img, raw_img, cam, mask):
        img_ = self.forward(img)
        raw_img_ = self.forward(raw_img)
        assert img_.size == raw_img_.size

        i, j, h, w = self.get_params(img_, self.size)
        cam_ = cam
        if cam_ is not None:
            cam_ = self.forward(cam_)
            cam_ = TF.crop(cam_, i, j, h, w)

        mask_ = mask
        if mask_ is not None:
            mask_ = self.forward(mask_)
            mask_ = TF.crop(mask_, i, j, h, w)

        return TF.crop(img_, i, j, h, w), TF.crop(raw_img_, i, j, h, w), cam_, mask_

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(
            self.size, self.padding)


class Resize(_BasicTransform):
    def __init__(self, size,
                 interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__()
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. "
                            "Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, "
                             "it should have 1 or 2 values")
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, raw_img, cam, mask):
        cam_ = cam
        if cam_ is not None:
            cam_ = TF.resize(cam_, self.size, self.interpolation)

        mask_ = mask
        if mask_ is not None:
            mask_ = TF.resize(mask_, self.size, self.interpolation)
            # mask_ = TF.resize(mask_, self.size, TF.InterpolationMode.NEAREST)

        return TF.resize(img, self.size, self.interpolation), TF.resize(
            raw_img, self.size, self.interpolation), cam_, mask_

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)