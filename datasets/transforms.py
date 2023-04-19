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

    def __call__(self, img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1, mask4, mask3, mask2, mask1):
        for t in self.transforms:
            if isinstance(t, (RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, Resize)):
                img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1, mask4, mask3, mask2, mask1 = \
                    t(img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1, mask4, mask3, mask2, mask1)
            else:
                img = t(img)

        return img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1, mask4, mask3, mask2, mask1

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class _BasicTransform(object):
    def __call__(self, img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1,
                 mask4, mask3, mask2, mask1):
        raise NotImplementedError


class RandomHorizontalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1,
                 mask4, mask3, mask2, mask1):
        if random.random() < self.p:

            std_cam4_ = std_cam4
            if std_cam4_ is not None:
                std_cam4_ = TF.hflip(std_cam4)

            std_cam3_ = std_cam3
            if std_cam3_ is not None:
                std_cam3_ = TF.hflip(std_cam3)

            std_cam2_ = std_cam2
            if std_cam2_ is not None:
                std_cam2_ = TF.hflip(std_cam2)

            std_cam1_ = std_cam1
            if std_cam1_ is not None:
                std_cam1_ = TF.hflip(std_cam1)

            mask4_ = mask4
            if mask4_ is not None:
                mask4_ = TF.hflip(mask4_)

            mask3_ = mask3
            if mask3_ is not None:
                mask3_ = TF.hflip(mask3_)

            mask2_ = mask2
            if mask2_ is not None:
                mask2_ = TF.hflip(mask2_)

            mask1_ = mask1
            if mask1_ is not None:
                mask1_ = TF.hflip(mask1_)

            return TF.hflip(img), TF.hflip(raw_img), std_cam4_, std_cam3_, std_cam2_, std_cam1_,\
                   mask4_, mask3_, mask2_, mask1_

        return img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1, mask4, mask3, mask2, mask1

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(_BasicTransform):
    def __init__(self, p=PROB_THRESHOLD):
        self.p = p

    def __call__(self, img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1,
                 mask4, mask3, mask2, mask1):
        if random.random() < self.p:

            std_cam4_ = std_cam4
            if std_cam4_ is not None:
                std_cam4_ = TF.vflip(std_cam4)

            std_cam3_ = std_cam3
            if std_cam3_ is not None:
                std_cam3_ = TF.vflip(std_cam3)

            std_cam2_ = std_cam2
            if std_cam2_ is not None:
                std_cam2_ = TF.vflip(std_cam2)

            std_cam1_ = std_cam1
            if std_cam1_ is not None:
                std_cam1_ = TF.vflip(std_cam1)

            mask4_ = mask4
            if mask4_ is not None:
                mask4_ = TF.vflip(mask4_)

            mask3_ = mask3
            if mask3_ is not None:
                mask3_ = TF.vflip(mask3_)

            mask2_ = mask2
            if mask2_ is not None:
                mask2_ = TF.vflip(mask2_)

            mask1_ = mask1
            if mask1_ is not None:
                mask1_ = TF.vflip(mask1_)

            return TF.vflip(img), TF.vflip(raw_img), std_cam4_, std_cam3_, std_cam2_, std_cam1_, \
                   mask4_, mask3_, mask2_, mask1_

        return img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1, mask4, mask3, mask2, mask1

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

    def __call__(self, img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1,
                 mask4, mask3, mask2, mask1):
        img_ = self.forward(img)
        raw_img_ = self.forward(raw_img)
        assert img_.size == raw_img_.size

        i, j, h, w = self.get_params(img_, self.size)
        std_cam4_ = std_cam4
        if std_cam4_ is not None:
            std_cam4_ = self.forward(std_cam4)
            std_cam4_ = TF.crop(std_cam4_, i, j, h, w)

        std_cam3_ = std_cam3
        if std_cam3_ is not None:
            std_cam3_ = self.forward(std_cam3)
            std_cam3_ = TF.crop(std_cam3_, i, j, h, w)

        std_cam2_ = std_cam2
        if std_cam2_ is not None:
            std_cam2_ = self.forward(std_cam2)
            std_cam2_ = TF.crop(std_cam2_, i, j, h, w)

        std_cam1_ = std_cam1
        if std_cam1_ is not None:
            std_cam1_ = self.forward(std_cam1)
            std_cam1_ = TF.crop(std_cam1_, i, j, h, w)

        mask4_ = mask4
        if mask4 is not None:
            mask4_ = self.forward(mask4_)
            mask4_ = TF.crop(mask4_, i, j, h, w)

        mask3_ = mask3
        if mask3 is not None:
            mask3_ = self.forward(mask3_)
            mask3_ = TF.crop(mask3_, i, j, h, w)

        mask2_ = mask2
        if mask2 is not None:
            mask2_ = self.forward(mask2_)
            mask2_ = TF.crop(mask2_, i, j, h, w)

        mask1_ = mask1
        if mask1 is not None:
            mask1_ = self.forward(mask1_)
            mask1_ = TF.crop(mask1_, i, j, h, w)

        return TF.crop(img_, i, j, h, w), TF.crop(
            raw_img_, i, j, h, w), std_cam4_, std_cam3_, std_cam2_, std_cam1_, mask4_, mask3_, mask2_, mask1_

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

    def __call__(self, img, raw_img, std_cam4, std_cam3, std_cam2, std_cam1,
                 mask4, mask3, mask2, mask1):
        std_cam4_ = std_cam4
        if std_cam4_ is not None:
            std_cam4_ = TF.resize(std_cam4_, self.size, self.interpolation)

        std_cam3_ = std_cam3
        if std_cam3_ is not None:
            std_cam3_ = TF.resize(std_cam3_, self.size, self.interpolation)

        std_cam2_ = std_cam2
        if std_cam2_ is not None:
            std_cam2_ = TF.resize(std_cam2_, self.size, self.interpolation)

        std_cam1_ = std_cam1
        if std_cam1_ is not None:
            std_cam1_ = TF.resize(std_cam1_, self.size, self.interpolation)

        mask4_ = mask4
        if mask4_ is not None:
            mask4_ = TF.resize(mask4_, self.size, self.interpolation)

        mask3_ = mask3
        if mask3_ is not None:
            mask3_ = TF.resize(mask3_, self.size, self.interpolation)

        mask2_ = mask2
        if mask2_ is not None:
            mask2_ = TF.resize(mask2_, self.size, self.interpolation)

        mask1_ = mask1
        if mask1_ is not None:
            mask1_ = TF.resize(mask1_, self.size, self.interpolation)

        return TF.resize(img, self.size, self.interpolation), TF.resize(
            raw_img, self.size, self.interpolation), std_cam4_, std_cam3_, std_cam2_, std_cam1_, \
               mask4_, mask3_,  mask2_, mask1_

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)