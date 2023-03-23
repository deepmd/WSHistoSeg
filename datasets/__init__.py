from datasets.transforms import Compose, Resize, RandomVerticalFlip, RandomHorizontalFlip, RandomCrop
from torchvision import transforms


def get_train_transforms(opt):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return Compose([
            Resize((opt.resize_size, opt.resize_size)),
            RandomCrop(opt.crop_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.5,
                                   saturation=0.5, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def get_val_transforms(opt):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return Compose([
            Resize((opt.crop_size, opt.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])


def get_transforms(opt):
    return {'train': get_train_transforms(opt),
            'valcl': get_val_transforms(opt),
            'test': get_val_transforms(opt)}
