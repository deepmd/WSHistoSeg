import os
import argparse
import random
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.transforms import Compose, Resize
from datasets.wsol_loaders import WsolDataset
from networks.resnet import resnet_encoders
from networks.aspp import ASPP
from networks.modules import BNReLU, ProjectionHead
from evaluation import MaskEvaluation
from utils import is_required_grad, get_cam_extractor, set_up_logger

encoders = {}
encoders.update(resnet_encoders)


class WGAP(torch.nn.Module):
    """ https://arxiv.org/pdf/1512.04150.pdf """
    def __init__(self, in_channels, classes):
        super(WGAP, self).__init__()
        self.name = 'WGAP'

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(in_channels, classes)

    @property
    def builtin_cam(self):
        return False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.reshape(pre_logit.size(0), -1)
        logits = self.fc(pre_logit)

        return logits


class STDCLModel(torch.nn.Module):
    def __init__(self, encoder_name, num_classes, depth=5, proj_dim=128, use_aspp=False):
        super().__init__()

        encoder = encoders[encoder_name]['encoder']
        encoder_params = encoders[encoder_name]['params']
        encoder_params.update(depth=depth)
        self.encoder = encoder(**encoder_params)

        self.classification_head = WGAP(encoder_params['out_channels'][-1], num_classes)

        self.use_aspp = use_aspp
        in_channels = encoder_params['out_channels'][-1]
        if self.use_aspp:
            self.aspp = ASPP(encoder_name, output_stride=8, BatchNorm=torch.nn.BatchNorm2d)
            self.seg_head = torch.nn.Sequential(
                torch.nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0, bias=False),
                BNReLU(256, bn_type='torchbn'),
                torch.nn.Dropout2d(0.10),
                torch.nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.proj_head = ProjectionHead(dim_in=1280, proj_dim=proj_dim, proj='linear')
        else:
            self.seg_head = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                BNReLU(in_channels, bn_type='torchbn'),
                torch.nn.Dropout2d(0.10),
                torch.nn.Conv2d(in_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.proj_head = ProjectionHead(dim_in=in_channels, proj_dim=proj_dim, proj='linear')

    def forward(self, x):
        features = self.encoder(x)
        cl_logits = self.classification_head(features[-1])
        x = features[-1]
        if self.use_aspp:
            x = self.aspp(x)
        seg = self.seg_head(x)
        embed = self.proj_head(x)
        return {'seg': seg, 'embed': embed, 'logits': cl_logits}


def create_model(cfg):
    model = STDCLModel(encoder_name='resnet50', num_classes=2, proj_dim=128, use_aspp=True)

    if cfg.encoder_weights_dir is not None:
        if 'cam' in cfg.method:
            encoder_weights = os.path.join(cfg.encoder_weights_dir, 'encoder.pt')
            classifier_weights = os.path.join(cfg.encoder_weights_dir, 'classification_head.pt')

            encoder_state_dict = torch.load(encoder_weights, map_location='cpu')
            classifier_state_dict = torch.load(classifier_weights, map_location='cpu')

            model.encoder.load_state_dict(encoder_state_dict, strict=True)
            model.classification_head.load_state_dict(classifier_state_dict, strict=True)
        else:
            weights = os.path.join(cfg.encoder_weights_dir, 'ckpt_test_rnd_4_iter_120_85.76070359312466.pth')
            state_dict = torch.load(weights)
            model.load_state_dict(state_dict['model'], strict=False)

    return model


def build_transforms(crop_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transforms = Compose([
        Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return val_transforms


def build_loaders(cfg):
    data_transforms = build_transforms(cfg.crop_size)
    dataset = WsolDataset(data_root=cfg.data_root,
                          metadata_root=os.path.join(cfg.metadata_root, cfg.split),
                          suffix=[],
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
    os.makedirs(os.path.join(cfg.run_dir, cfg.method, cfg.split), exist_ok=True)
    logger = set_up_logger(logs_path=os.path.join(cfg.run_dir, cfg.method),
                           log_file_name=f'sample_metrics_{cfg.split}')
    # set deterministic training for reproducibility
    cfg.seed = 0
    torch.manual_seed(cfg.seed)  # seed the RNG for all devices (both CPU and CUDA)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cpu")
    if torch.cuda.is_available() is not None:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # set to False due to performance issues

    data_loader = build_loaders(cfg)
    model = create_model(cfg).to(device)

    cam_extractor = None
    if 'cam' in cfg.method:
        trg_layer = 'encoder.layer4.2.relu3'
        fc_layer = 'classification_head.fc'
        cam_extractor = get_cam_extractor(cfg.method, model, trg_layer, fc_layer)

    model.eval()
    split_evaluator = MaskEvaluation(cam_curve_interval=0.001)
    with torch.set_grad_enabled(is_required_grad(cfg.method)):
        for idx, batch_data in enumerate(data_loader):
            sample_evaluator = MaskEvaluation(cam_curve_interval=0.001)
            image = batch_data['image'].to(device)
            gt_mask = batch_data['mask']
            label = batch_data['label'].item()
            image_name = batch_data['image_id'][0].split('/')[-1]

            out = model(image)
            if 'cam' in cfg.method and cam_extractor is not None:
                # Retrieve the CAM by passing the class index and the model outpu
                activation_map = cam_extractor(label, out['logits'])
                with torch.no_grad():
                    cam = activation_map[0].squeeze(0)
                    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                        image.shape[2:],
                                        mode='bilinear',
                                        align_corners=True).squeeze(0).squeeze(0)
                    cam = cam.cpu().numpy().astype(float)
            else:
                logits_seg = out['seg']
                logits_seg = F.interpolate(logits_seg, image.shape[2:],
                                           mode='bicubic',
                                           align_corners=False)
                cam = torch.sigmoid(logits_seg[:, 1]).squeeze(0).cpu().numpy().astype(float)
                # cam = torch.softmax(logits_seg, dim=1)[:, 1].squeeze(0).cpu().numpy().astype(float)
            if cfg.save_cams:
                np.save(os.path.join(cfg.run_dir, cfg.method, cfg.split, image_name.split('.')[0] + '.npy'), cam)
            mask = F.interpolate(gt_mask.float(), image.shape[2:], mode='nearest').squeeze(0).squeeze(0)
            mask = mask.cpu().numpy().astype(np.uint8)

            split_evaluator.accumulate(cam, mask)
            sample_evaluator.accumulate(cam, mask)
            sample_evaluator.compute()
            sample_metrics = sample_evaluator.perf_gist
            log_messages = [f'\n{image_name}']
            for key, value in sample_metrics.items():
                if not isinstance(value, np.ndarray):
                    log_messages.append(f'{key}: {value}')
            logger.info('\n'.join(log_messages))

            print(f'{idx + 1}/{len(data_loader)}')

    split_evaluator.compute()
    split_metrics = split_evaluator.perf_gist

    if cfg.save_metrics:
        np.save(os.path.join(cfg.run_dir, cfg.method, f'split_metrics_{cfg.split}_{cfg.method}.npy'), split_metrics)

    with open(os.path.join(cfg.run_dir, cfg.method, f'split_metrics_{cfg.split}.txt'), 'w') as f:
        for key, value in split_metrics.items():
            if not isinstance(value, np.ndarray):
                f.write(f"{key}: {value}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root', type=str, default="/home/reza/Documents/GLAS", help='')
    parser.add_argument('--metadata_root', type=str, default="../datasets/folds/GLAS/fold-0", help='')
    parser.add_argument('--method', type=str, default='ours',
                        choices=['gradcam', 'ours', 'gradcampp', 'smoothgradcampp',
                                 'xgradcam', 'layercam', 'cam', 'scorecam', 'iscam'],
                        help='')
    parser.add_argument('--split', type=str, default='test', help='')
    parser.add_argument('--run_dir', type=str, default="cams", help='')
    parser.add_argument('--encoder_weights_dir', type=str, default="../weights", help='')
    parser.add_argument('--crop_size', type=int, default=224, help='')
    parser.add_argument('--num_workers', type=int, default=0, help='')
    parser.add_argument('--save_cams', action='store_true', help='')
    parser.add_argument('--save_metrics', action='store_true', help='')
    args = parser.parse_args()
    main(args)