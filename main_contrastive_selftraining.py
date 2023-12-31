import glob
import os
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import Optional
from torch import Tensor

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import set_up_logger, log_parameters, save_checkpoint
from datasets import get_transforms
from optim_scheduler import get_optim_scheduler
from losses import ContrastCELoss
from utils import AverageMeter, save_pseudo_labels
# from metrics import ConfMatrix

from datasets.wsol_loaders import WsolDataset
from networks import create_model
from evaluation import MaskEvaluation


def parse_options():
    parser = argparse.ArgumentParser('arguments for training')

    # model, dataset
    parser.add_argument('--data_root', type=str, default=None, help='path to dataset')
    parser.add_argument('--task', type=str, default='stdcl', choices=['stdcl', 'unet'])
    parser.add_argument('--encoder_name', type=str, default='resnet50', choices=['resnet50'])
    parser.add_argument('--proj_dim', type=int, default=128, help='num of channels in output of projection head')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes.')
    parser.add_argument('--pretrained', type=str, default=None, help='path to pretrained weights.')
    parser.add_argument('--use_pseudo_mask', dest="use_pseudo_mask", action='store_true',
                        help='using pseudo masks for training.')
    parser.add_argument("--use_aspp", dest="use_aspp", action="store_true",
                        help="use aspp module at the esnd of encoder.")

    # optimization
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lr_policy', type=str, default='lambda_poly', help='scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_heads_ratio', type=int, default=10, help="learning rate ratio for adjusting heads' lr")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate.")
    parser.add_argument("--loss_weight", type=float, default=0.1, help="the weight is used for balancing losses.")
    parser.add_argument('--gamma', type=float, default=2, help='Gamma value for reverse focal loss (default: 2)')

    # contrastive loss
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature in contrastive loss.")
    parser.add_argument("--base_temperature", type=float, default=0.07, help="base temperature in contrastive loss.")
    parser.add_argument('--num_samples', type=int, default=10, help='max samples')

    # train settings
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--max_iters', type=int, default=3000, help='max epochs for training.')
    parser.add_argument('--num_rounds', type=int, default=5, help='max epochs for training.')
    parser.add_argument('--pseudo_labeling_step', type=int, default=200, help='')

    parser.add_argument('--resize_size', type=int, default=256, help='resize size')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size')
    parser.add_argument('--metadata_root', type=str, help='path to related fold')

    # # other setting
    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--eval_freq', type=int, default=30, help='evaluation model on validation set frequency')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument("--debug", dest="debug", action="store_true", help="activate debugging mode")

    opt = parser.parse_args()

    opt.ce_weights = [1.0, 1.0]
    opt.ignore_index = 255
    # opt.pseudo_labeling_step = int(opt.max_iters / opt.num_rounds) + 1

    opt.model_name = f"glas_model_{opt.encoder_name}_{opt.optimizer}" + \
                     f"_lr_{opt.learning_rate}_bsz_{opt.batch_size}_loss_CE_CL_trial_{opt.trial}"

    save_path = os.path.join("./save", opt.model_name)

    opt.tb_folder = os.path.join(save_path, "tensorboard")
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(save_path, "models")
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.log_folder = os.path.join(save_path, "logs")
    if not os.path.isdir(opt.log_folder):
        os.makedirs(opt.log_folder)

    return opt


def set_loader(opt):
    logger = opt.logger
    data_transforms = get_transforms(opt)

    datasets = {
        split: WsolDataset(
            data_root=opt.data_root,
            metadata_root=os.path.join(opt.metadata_root, split if split != 'train_ps' else 'train'),
            transforms=data_transforms[split]
        )
        for split in ['train', 'valcl', 'test', 'train_ps']
    }

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(opt.seed)

    data_loaders = {
        split: DataLoader(
            dataset=datasets[split],
            batch_size=opt.batch_size if split == 'train' else 1,
            shuffle=True if split == 'train' else False,
            num_workers=opt.num_workers,
            worker_init_fn=seed_worker,
            generator=g
        )
        for split in ['train', 'valcl', 'test', 'train_ps']
    }
    # pin_memory = True,
    # drop_last = True,

    logger.info(f"Summary of the data:")
    logger.info(f"Number of images in training set = {len(datasets['train'])}")
    logger.info(f"Number of images in validation set = {len(datasets['valcl'])}")

    return data_loaders


def set_model(opt):
    model = create_model(opt).to(opt.device)
    criterion = ContrastCELoss(opt.ce_weights, opt.ignore_index, opt.loss_weight,
                               opt.temperature, opt.base_temperature, opt.num_samples,
                               d_fg=0.996, d_bg=0.999, gamma=opt.gamma)
    optimizer, scheduler = get_optim_scheduler(model, opt)

    return model, criterion, optimizer, scheduler


@torch.no_grad()
def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
    """CAM normalization."""
    spatial_dims = cams.ndim - 1 if spatial_dims is None else spatial_dims
    cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
    cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

    return cams


def evaluate(model, val_loader, criterion, opt, pseudo_labels_path=None):
    model.eval()
    criterion.eval()

    # metrics = ConfMatrix(opt.num_classes)
    evaluator = MaskEvaluation(cam_curve_interval=0.001)

    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(val_loader)):
            images = data_dict['image'].to(opt.device)
            gt_masks = data_dict['mask'].to(opt.device)

            logits = model(images)
            logits_seg = logits['seg']
            if list(logits_seg.shape[2:]) != list(gt_masks.shape[1:]):
                logits_seg = F.interpolate(logits_seg, gt_masks.size()[2:],
                                           mode='bilinear', align_corners=True)
            # logits_seg = torch.sigmoid(logits_seg[:, 1]).squeeze(0).detach().cpu().numpy().astype(float)
            logits_seg = torch.softmax(logits_seg, dim=1)[:, 1].squeeze(0).detach().cpu().numpy().astype(float)
            if pseudo_labels_path is not None:
                np.save(pseudo_labels_path, logits_seg)
            mask = gt_masks.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            evaluator.accumulate(logits_seg, mask)

        pxap = evaluator.compute()
        results = evaluator.perf_gist

    return results


def train(model, criterion, data_loaders, optimizer, scheduler, opt, round):
    model.train()
    criterion.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    contrast_losses = AverageMeter()
    expand_losses = AverageMeter()

    end = time.time()
    total_corrects = [0, 0]
    for data_dict in data_loaders['train']:
        data_time.update(time.time() - end)

        bsz = data_dict['image'].size(0)
        images = data_dict['image'].to(opt.device)
        labels = data_dict['label'].to(opt.device)
        masks = data_dict['mask'].to(opt.device)
        cams = data_dict['cam']

        # compute loss
        outputs = model(images)
        loss, partial_losses, num_corrects = criterion(outputs, cams, masks, labels,
                                                       [0.2, 0.3, 0.3, 0.4, 0.5][round-1], opt.use_pseudo_mask)
        total_corrects[0] += num_corrects[0]
        total_corrects[1] += num_corrects[1]

        with torch.no_grad():
            losses.update(loss.item(), bsz)
            ce_losses.update(partial_losses['ce'], bsz)
            contrast_losses.update(partial_losses['contrast'], bsz)
            expand_losses.update(partial_losses['expand'], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(opt.current_iter)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # updating num of iterations till now
        opt.current_iter += 1

        # add info to tensorboard
        opt.tb_logger.add_scalar('train/total_loss', loss, opt.current_iter)
        opt.tb_logger.add_scalar('train/ce_loss', partial_losses['ce'], opt.current_iter)
        opt.tb_logger.add_scalar('train/contrast_loss', partial_losses['contrast'], opt.current_iter)
        opt.tb_logger.add_scalar('train/learning_rate_groups0', optimizer.param_groups[0]['lr'], opt.current_iter)
        opt.tb_logger.add_scalar('train/learning_rate_groups1', optimizer.param_groups[1]['lr'], opt.current_iter)

        # print info
        if opt.current_iter % opt.print_freq == 0:
            opt.logger.info(f"[Train] [Epoch {opt.current_epoch}] " +
                            f"[Iteration {opt.current_iter}/{opt.max_iters}] " +
                            f"BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t" +
                            f"DT {data_time.val:.3f} ({data_time.avg:.3f})\t" +
                            f"Total Loss: {losses.val:.3f} ({losses.avg:.3f}) " +
                            f"CE Loss: {ce_losses.val:.3f} ({ce_losses.avg:.3f}) " +
                            f"Contrast Loss: {contrast_losses.val:.3f} ({contrast_losses.avg:.3f})\t" +
                            f"Expand Loss: {expand_losses.val:.3f} ({expand_losses.avg:.3f})\t" +
                            "Learning rate: {}".format([param_group['lr'] for param_group in optimizer.param_groups]))

        if opt.current_iter % opt.eval_freq == 0 or opt.current_iter % opt.max_iters == 0:
            for split in ['valcl', 'test']:
                metrics = evaluate(model, data_loaders[split], criterion, opt)
                opt.logger.info(f"[{split.upper()}] [Epoch {opt.current_epoch}] "
                                f"[Iteration {opt.current_iter}]\t")
                for metric, value in metrics.items():
                    opt.logger.info(f"{metric}: {value}")

                # add info to tensorboard
                opt.tb_logger.add_scalar(f'{split}/PXAP', metrics['PXAP'], opt.current_iter)

                if metrics['PXAP'] > opt.best_val_pxap[split]:
                    opt.best_val_pxap[split] = metrics['PXAP']
                    # save model
                    filename = f"ckpt_{split}_iter_{opt.current_iter}_{metrics['PXAP']}.pth"
                    save_file = os.path.join(opt.save_folder, filename)
                    if opt.current_iter % opt.max_iters == 0:
                        save_file = os.path.join(opt.save_folder, f"last_{metrics['PXAP']}.pth")
                    save_checkpoint(model, optimizer, opt, opt.current_iter, save_file)

                opt.logger.info(f"[BEST PXAP on {split.upper()}={opt.best_val_pxap[split]}]")

            # changing the phase of the model to train
            model.train()
            criterion.train()

        if opt.current_iter % opt.max_iters == 0:
            break

    opt.current_epoch += 1

    return losses.avg, ce_losses.avg, contrast_losses.avg, total_corrects


def main():
    opt = parse_options()

    # Set deterministic training for reproducibility
    opt.seed = 0
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    opt.device = torch.device('cpu')
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    # logger
    logger = set_up_logger(logs_path=opt.log_folder)
    log_parameters(opt, logger)
    opt.logger = logger

    # tensorboard
    tb_logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=30)
    opt.tb_logger = tb_logger

    # build data loader
    data_loaders = set_loader(opt)

    # build model, criterion and optimizer and scheduler
    model, criterion, optimizer, scheduler = set_model(opt)
    model = model.to(opt.device)
    criterion = criterion.to(opt.device)

    # training routine
    opt.current_epoch = 1
    opt.current_iter = 1
    opt.best_val_pxap = {'valcl': -1, 'test': -1}
    round = 1
    while opt.current_iter < opt.max_iters:
        # opt.current_iter and opt.current_epoch are updated in the train function.
        time1 = time.time()
        loss, ce_loss, contrast_loss, total_corrects = train(model, criterion, data_loaders,
                                                             optimizer, scheduler, opt, round)
        time2 = time.time()
        CoFG = total_corrects[0] / (len(data_loaders['train'].dataset) * opt.num_samples) * 100
        CoBG = total_corrects[1] / (len(data_loaders['train'].dataset) * opt.num_samples) * 100
        opt.logger.info(f"[End Epoch {opt.current_epoch - 1}] Train Time: {(time2 - time1):0.2f}, " +
                        f"Loss: {loss:06.3f} (CE: {ce_loss:06.3f}, Contrast: {contrast_loss:06.3f}) " +
                        f"CoFG: {CoFG:.3f}, CoBG: {CoBG:.3f}")
        opt.tb_logger.add_scalar('train/total_loss_avg', loss, opt.current_epoch)
        opt.tb_logger.add_scalar('train/ce_loss_avg', ce_loss, opt.current_epoch)
        opt.tb_logger.add_scalar('train/contrast_loss_avg', contrast_loss, opt.current_epoch)
        opt.tb_logger.add_scalar('train/CoFG', CoFG, opt.current_epoch)
        opt.tb_logger.add_scalar('train/CoBG', CoBG, opt.current_epoch)

        if opt.current_epoch % opt.pseudo_labeling_step == 0:
            if round != 5:
                model_ = create_model(opt).to(opt.device)
                # Loading best model in previous iterations
                checkpoint_path = glob.glob(os.path.join(opt.save_folder,
                f"ckpt_test_*{opt.best_val_pxap['test']}.pth"))[0]
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                model_.load_state_dict(state_dict['model'])
                # Generating CAMs
                save_path = os.path.join(opt.data_root, 'Warwick_QU_Dataset_(Released_2016_07_08)/CAMs/Layer4')
                save_pseudo_labels(model_, data_loaders, save_path, round, opt.logger, opt.device)
                # round += 1
                # Restarting optimizer and scheduler to initial state
                # opt.current_epoch = 1
                # opt.current_iter = 1
                # model, ـ, optimizer, scheduler = set_model(opt)
            break

    opt.logger.info(f"[End of training]:\n "
                    f"BEST PXAP on VALCL={opt.best_val_pxap['valcl']} \n "
                    f"BEST PXAP on TEST={opt.best_val_pxap['test']}")


if __name__ == '__main__':
    main()