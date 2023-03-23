import argparse
import os
import random
import numpy as np
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from PIL import Image


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image, normalize
import torch.nn.functional as F

from networks import create_model
from datasets.wsol_loaders import WsolDataset
from datasets.transforms import Resize, Compose, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from utils import configure_metadata, get_cam_extractor, is_required_grad


def build_transforms(cfg):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # train_transforms = Compose([
    #         Resize((cfg.resize_size, cfg.resize_size)),
    #         RandomCrop(cfg.crop_size),
    #         RandomHorizontalFlip(),
    #         RandomVerticalFlip(),
    #         transforms.ColorJitter(brightness=0.5, contrast=0.5,
    #                                saturation=0.5, hue=0.05),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean, std)
    #     ])
    val_transforms = Compose([
            Resize((cfg.crop_size, cfg.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # return train_transforms if cfg.split == 'train' else val_transforms
    return val_transforms


def build_loaders(cfg):
    data_transforms = build_transforms(cfg)
    dataset = WsolDataset(data_root=cfg.data_root,
                          metadata_root=os.path.join(cfg.metadata_root, cfg.split),
                          transforms=data_transforms)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             worker_init_fn=seed_worker,
                             generator=g)

    return data_loader


def main(cfg):
    os.makedirs(cfg.run_dir, exist_ok=True)
    # set deterministic training for reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(cfg.seed)  # seed the RNG for all devices (both CPU and CUDA)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cpu")
    if torch.cuda.is_available() is not None:
        device = torch.device("cuda")
        # torch.cuda.set_device()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True  # set to False due to performance issues

    data_loader = build_loaders(cfg)
    model = create_model(cfg).to(device)
    trg_layer = 'encoder.layer4.2.relu3'
    fc_layer = 'classification_head.fc'
    cam_extractor = get_cam_extractor(cfg.method, model, trg_layer, fc_layer)

    for idx, batch_data in enumerate(data_loader):
        model.eval()
        image = batch_data['image'].to(device)
        gt_mask = batch_data['mask']
        label = batch_data['label'].item()
        image_name = batch_data['image_id'][0].split('/')[-1]
        out_path = os.path.join(cfg.run_dir, cfg.method, 'cams', image_name + '.npy')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # image_path = os.path.join(cfg.data_root, batch_data['image_id'][0])
        # img = Image.open(image_path)
        # img = img.convert('RGB')  # H, W, 3
        # img = image.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        # img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        # img *= 255
        # img = image.squeeze(0).transpose(0, 2).cpu().numpy()
        img = image.squeeze(0).cpu().numpy().transpose((1, 2, 0))
        img = np.array([0.229, 0.224, 0.225]) * img + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1) * 255
        img = img.astype(np.uint8)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

        with torch.set_grad_enabled(is_required_grad(cfg.method)):
            # Preprocess your data and feed it to the model
            out = model(image)

            # Retrieve the CAM by passing the class index and the model outpu
            activation_map = cam_extractor(label, out)
            np.save(out_path, activation_map[0].squeeze(0).cpu().numpy())

            # Visualize the raw CAM
            # plt.imshow(activation_map[0].squeeze(0).cpu().numpy())
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()

            # Resize the CAM and overlay it
            result, edited_cam = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            out_path = os.path.join(cfg.run_dir, cfg.method, 'overlay_images', image_name + '_CamOnImage.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            result.save(out_path)

            # # Visualize the raw CAM
            # plt.imshow(edited_cam);
            # plt.axis('off');
            # plt.tight_layout();
            # plt.show()

            # Display it
            # plt.imshow(result)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()

            # gt_mask = gt_mask.squeeze(0).numpy()
            gt_mask = F.interpolate(gt_mask.float(), image.shape[2:], mode='nearest').squeeze(0).squeeze(0)
            gt_mask = torch.stack([gt_mask, gt_mask, gt_mask])
            gt_mask = to_pil_image(gt_mask, mode='RGB')
            # gt_mask = gt_mask.numpy().squeeze(0).squeeze(0) * 255

            # Display it
            # plt.imshow(gt_mask)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()

            # plt.imshow(image.cpu().numpy().squeeze(0).transpose((1, 2, 0)))
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()

            # final_result = Image.blend(result, gt_mask, 0.5)
            # Overlay the image with the mask
            alpha = 0.7
            final_result = Image.fromarray((alpha * np.asarray(result) + (1 - alpha) * np.asarray(gt_mask)).astype(np.uint8))
            out_path = os.path.join(cfg.run_dir, cfg.method, 'overlay_images', image_name + '_CamMaskOnImage.png')
            final_result.save(out_path)

            # plt.imshow(final_result)
            # plt.axis('off')
            # plt.tight_layout()
            # plt.show()
        print(f'{idx+1}/{len(data_loader)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default="/home/reza/Documents/GLAS", help='')
    parser.add_argument('--metadata_root', type=str, default="/home/reza/Documents/Earasing/folds/GLAS/fold-0", help='')
    parser.add_argument('--method', type=str, default='gradcam', help='')
    parser.add_argument('--split', type=str, default='train', help='')
    parser.add_argument('--run_dir', type=str, default="Grad_CAMs/train", help='')
    parser.add_argument('--encoder_name', type=str, default="resnet50", help='')
    parser.add_argument('--encoder_weights_dir', type=str, default="weights", help='')
    parser.add_argument('--resize_size', type=int, default=256, help='')
    parser.add_argument('--crop_size', type=int, default=224, help='')
    parser.add_argument('--num_classes', type=int, default=2, help='')
    # parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--num_workers', type=int, default=1, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    args = parser.parse_args()

    # c = 1
    # run_dir = args.run_dir
    # while os.path.exists(run_dir):
    #     run_dir = args.run_dir + f'{c}'
    #     c += 1
    # args.run_dir = run_dir

    main(args)



