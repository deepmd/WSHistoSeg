import argparse
import os
import random
import numpy as np
from torchcam.utils import overlay_mask
from PIL import Image


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

# from networks import create_model
from datasets.wsol_loaders import WsolDataset
from evaluation import MaskEvaluation
from datasets.transforms import Resize, Compose
from utils import get_cam_extractor, is_required_grad
from networks.STDCLModel import STDCLModel
import networks.initialization as init
from networks.modules import BNReLU, ProjectionHead
from networks.resnet import resnet_encoders
from networks.aspp import ASPP


encoders = {}
encoders.update(resnet_encoders)


class WGAP(nn.Module):
    """ https://arxiv.org/pdf/1512.04150.pdf """
    def __init__(self, in_channels, classes):
        super(WGAP, self).__init__()
        self.name = 'WGAP'

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, classes)

    @property
    def builtin_cam(self):
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        return logits


# class STDCLModel(nn.Module):
#     def __init__(self, encoder_name, num_classes, method):
#         super().__init__()
#
#         self.method = method
#         encoder = encoders[encoder_name]['encoder']
#         encoder_params = encoders[encoder_name]['params']
#         self.encoder = encoder(**encoder_params)
#
#         in_channels = encoder_params['out_channels'][-1]
#         if 'cam' in self.method:
#             self.classification_head = WGAP(in_channels, num_classes)
#             self.initialize()
#         else:
#             self.seg_head = nn.Sequential(
#                 nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
#                 BNReLU(in_channels, bn_type='torchbn'),
#                 nn.Dropout2d(0.10),
#                 nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
#             )
#
#     def initialize(self):
#         if self.classification_head is not None:
#             init.initialize_head(self.classification_head)
#
#     def forward(self, x):
#         features = self.encoder(x)
#         if 'cam' in self.method:
#             out = self.classification_head(features[-1])
#         else:
#             out = self.seg_head(features[-1])
#         return out

class STDCLModel(nn.Module):
    def __init__(self, encoder_name, num_classes, depth=5, proj_dim=128, use_aspp=False):
        super().__init__()

        encoder = encoders[encoder_name]['encoder']
        encoder_params = encoders[encoder_name]['params']
        encoder_params.update(depth=depth)
        self.encoder = encoder(**encoder_params)

        # self.classification_head = WGAP(encoder_params['out_channels'][-1], num_classes)

        self.use_aspp = use_aspp
        in_channels = encoder_params['out_channels'][-1]
        if self.use_aspp:
            self.aspp = ASPP(encoder_name, output_stride=8, BatchNorm=nn.BatchNorm2d)
            self.seg_head = nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=False),
                BNReLU(256, bn_type='torchbn'),
                nn.Dropout2d(0.10),
                nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.proj_head = ProjectionHead(dim_in=1280, proj_dim=proj_dim, proj='linear')
        else:
            self.seg_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                BNReLU(in_channels, bn_type='torchbn'),
                nn.Dropout2d(0.10),
                nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
            # self.seg_head = SegHead(in_channels, num_classes)
            self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')

        # self.initialize()

    def initialize(self):
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def forward(self, x):
        features = self.encoder(x)
        # cl_logits = self.classification_head(features[-1])
        x = features[-1]
        if self.use_aspp:
            x = self.aspp(x)
        seg = self.seg_head(x)
        embed = self.proj_head(x)
        return {'seg': seg, 'embed': embed}


def create_model(cfg):
    model = STDCLModel(encoder_name=cfg.encoder_name, num_classes=cfg.num_classes, proj_dim=128, use_aspp=True)

    if cfg.encoder_weights_dir is not None:
        if 'cam' in cfg.method:
            encoder_weights = os.path.join(cfg.encoder_weights_dir, 'encoder.pt')
            classifier_weights = os.path.join(cfg.encoder_weights_dir, 'classification_head.pt')

            encoder_state_dict = torch.load(encoder_weights, map_location='cpu')
            classifier_state_dict = torch.load(classifier_weights, map_location='cpu')

            model.encoder.load_state_dict(encoder_state_dict, strict=True)
            model.classification_head.load_state_dict(classifier_state_dict, strict=True)
        else:
            weights = os.path.join(cfg.encoder_weights_dir, 'ckpt_test_iter_60_85.32164703922639.pth')
            state_dict = torch.load(weights)
            model.load_state_dict(state_dict['model'], strict=False)

    return model


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

    cam_extractor = None
    if 'cam' in cfg.method:
        trg_layer = 'encoder.layer4.2.relu3'
        fc_layer = 'classification_head.fc'
        cam_extractor = get_cam_extractor(cfg.method, model, trg_layer, fc_layer)
    evaluator = MaskEvaluation(cam_curve_interval=0.001)

    for idx, batch_data in enumerate(data_loader):
        model.eval()
        image = batch_data['image'].to(device)
        gt_mask = batch_data['mask']
        label = batch_data['label'].item()
        image_name = batch_data['image_id'][0].split('/')[-1]
        out_path = os.path.join(cfg.run_dir, cfg.method, 'cams', image_name + '.npy')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

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

            if 'cam' in cfg.method and cam_extractor is not None:
                # Retrieve the CAM by passing the class index and the model outpu
                activation_map = cam_extractor(label, out)
                np.save(out_path, activation_map[0].squeeze(0).cpu().numpy())
                with torch.no_grad():
                    cam = activation_map[0].squeeze(0)
                    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                        image.shape[2:],
                                        mode='bilinear',
                                        align_corners=False).squeeze(0).squeeze(0)
                    cam = cam.detach().cpu().numpy().astype(float)
            else:
                assert out['seg'].size()[1] == cfg.num_classes
                cam = torch.sigmoid(out['seg'][:, 1])
                cam = cam.squeeze(0)
                np.save(out_path, cam.cpu().numpy())
                cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                    image.shape[2:],
                                    mode='bilinear',
                                    align_corners=False).squeeze(0).squeeze(0)
                cam = cam.detach().cpu().numpy().astype(float)

            mask = F.interpolate(gt_mask.float(), image.shape[2:], mode='nearest').squeeze(0).squeeze(0)
            mask = mask.cpu().numpy().astype(np.uint8)
            evaluator.accumulate(cam, mask)

            # Resize the CAM and overlay it
            mode = 'F' if 'cam' in cfg.method else None
            result, edited_cam = overlay_mask(to_pil_image(img), to_pil_image(cam.astype(np.float32), mode=mode), alpha=0.5)
            out_path = os.path.join(cfg.run_dir, cfg.method, 'overlay_images', image_name + '_CamOnImage.png')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            result.save(out_path)

            gt_mask = F.interpolate(gt_mask.float(), image.shape[2:], mode='nearest').squeeze(0).squeeze(0)
            gt_mask = torch.stack([gt_mask, gt_mask, gt_mask])
            gt_mask = to_pil_image(gt_mask, mode='RGB')

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

    AXAP = evaluator.compute()
    results = evaluator.perf_gist

    with open(os.path.join(cfg.run_dir, 'Stage1_Metrics.txt'), 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default="/home/reza/Documents/GLAS", help='')
    parser.add_argument('--metadata_root', type=str,
                        default="/home/reza/Documents/WSHistoSeg/datasets/folds/GLAS/fold-0", help='')
    parser.add_argument('--method', type=str, default='segmask', help='')
    parser.add_argument('--split', type=str, default='test', help='')
    parser.add_argument('--run_dir', type=str, default="segmask_85.32", help='')
    parser.add_argument('--encoder_name', type=str, default="resnet50", help='')
    parser.add_argument('--encoder_weights_dir', type=str, default="../weights", help='')
    parser.add_argument('--resize_size', type=int, default=256, help='')
    parser.add_argument('--crop_size', type=int, default=224, help='')
    parser.add_argument('--num_classes', type=int, default=2, help='')
    # parser.add_argument('--batch_size', type=int, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='')
    parser.add_argument('--seed', type=int, default=0, help='')
    args = parser.parse_args()

    # c = 1
    # run_dir = args.run_dir
    # while os.path.exists(run_dir):
    #     run_dir = args.run_dir + f'{c}'
    #     c += 1
    # args.run_dir = run_dir

    main(args)



