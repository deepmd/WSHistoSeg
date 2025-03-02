import torch
from torch import nn
import torch.nn.functional as F
from .utils import generate_foreground_background_mask, generate_pseudo_mask_by_cam, sample_bg, sample_foreground_background_mask


# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, class_weights, ignore_index, sample_ratio_ce):
        super(FSCELoss, self).__init__()
        weight = torch.FloatTensor(class_weights)
        self.ignore_index = ignore_index
        self.sample_ratio_ce = sample_ratio_ce
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index, reduction='none')

    def forward(self, inputs, cams, masks, use_pseudo_mask):
        target_mask = generate_pseudo_mask_by_cam(cams, self.ignore_index, self.sample_ratio_ce)
        target_mask = target_mask.to(inputs.device)
        target_mask = torch.where(use_pseudo_mask[:, None, None, None], target_mask, masks)

        target_mask = self._scale_target(target_mask, (inputs.size(2), inputs.size(3)))
        loss = self.ce_loss(inputs, target_mask).mean()

        masks = self._scale_target(masks, (inputs.size(2), inputs.size(3)))
        num_fg = (target_mask == 1).sum()
        num_bg = (target_mask == 0).sum()
        num_fg_corrects = int((masks[target_mask == 1] == 1).sum())
        num_bg_corrects = int((masks[target_mask == 0] == 0).sum())
        return loss, [num_fg_corrects, num_bg_corrects], [num_fg, num_bg]

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class ContrastCELoss(nn.Module):
    def __init__(self, class_weights, ignore_index, loss_weight,
                 temperature, base_temperature, d_fg, d_bg, gamma,
                 labeled_sample_ratio_cl, sample_ratio_cl, sample_ratio_ce):
        super(ContrastCELoss, self).__init__()

        self.loss_weight = loss_weight
        self.seg_criterion = FSCELoss(class_weights=class_weights,
                                      ignore_index=ignore_index,
                                      sample_ratio_ce=sample_ratio_ce)
        # self.seg_criterion = DynamicLoss(gamma, None, ignore_index, reduction='None')
        self.contrast_criterion = PixelContrastLoss(
            temperature=temperature, base_temperature=base_temperature, ignore_index=ignore_index,
            labeled_sample_ratio=labeled_sample_ratio_cl, sample_ratio=sample_ratio_cl
        )
        # self.expand_loss = ExpandLoss(d_fg, d_bg)

    def forward(self, preds, cams, masks, labels, use_pseudo_mask):
        metrics = {'pseudo_mask_ce': {'n_corrects': [], 'n_sampled': []},
                   'pseudo_mask_cl': {'n_corrects': [], 'n_sampled': []}}
        partial_losses = dict()

        assert "seg" in preds
        assert "embed" in preds

        seg_preds = preds['seg']
        embedding = preds['embed']

        loss_ce, num_corrects, num_sampled = self.seg_criterion(seg_preds, cams, masks, use_pseudo_mask)
        partial_losses['ce'] = loss_ce
        metrics['pseudo_mask_ce']['n_corrects'].extend(num_corrects)
        metrics['pseudo_mask_ce']['n_sampled'].extend(num_sampled)

        loss_contrast, num_corrects, num_sampled = \
            self.contrast_criterion(embedding, cams, masks, use_pseudo_mask, labels)
        partial_losses['contrast'] = loss_contrast
        metrics['pseudo_mask_cl']['n_corrects'].extend(num_corrects)
        metrics['pseudo_mask_cl']['n_sampled'].extend(num_sampled)

        # loss_expand = self.expand_loss(seg_preds)

        return loss_ce + self.loss_weight * loss_contrast, partial_losses, metrics


class PixelContrastLoss(nn.Module):
    def __init__(self, temperature, base_temperature, ignore_index, labeled_sample_ratio, sample_ratio):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.ignore_index = ignore_index
        self.labeled_sample_ratio = labeled_sample_ratio
        self.sample_ratio = sample_ratio

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]
        labels_ = labels_.contiguous().view(-1, 1)

        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(1,
                                                     torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
                                                     0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, embeddings, cams, masks, use_pseudo_mask, labels=None):
        h, w = embeddings.size()[2:]
        masks = F.interpolate(masks.float(), (h, w), mode='nearest')
        masks = masks.squeeze(1).byte()

        cams = F.interpolate(cams, (h, w), mode='bicubic', align_corners=True)
        cams = cams.squeeze(1)

        embeddings = embeddings.permute(0, 2, 3, 1)

        selected_pixels = generate_foreground_background_mask(cams, self.ignore_index, self.sample_ratio)
        selected_pixels = selected_pixels.to(masks.device)
        if 0 < self.labeled_sample_ratio < 1:
            masks = sample_foreground_background_mask(masks, self.ignore_index, self.labeled_sample_ratio)
        selected_pixels = torch.where(use_pseudo_mask[:, None, None], selected_pixels, masks)

        fg_feats = embeddings[selected_pixels == 1]
        fg_labels = torch.ones(fg_feats.shape[0])

        bg_feats = embeddings[selected_pixels == 0]
        bg_labels = torch.zeros(bg_feats.shape[0])

        all_feats = torch.cat([fg_feats, bg_feats], dim=0).unsqueeze(1)
        all_labels = torch.cat([fg_labels, bg_labels], dim=0)

        contrast_loss = self._contrastive(all_feats, all_labels)

        num_fg = (selected_pixels == 1).sum()
        num_bg = (selected_pixels == 0).sum()
        num_fg_corrects = int((masks[selected_pixels == 1] == 1).sum())
        num_bg_corrects = int((masks[selected_pixels == 0] == 0).sum())
        return contrast_loss, [num_fg_corrects, num_bg_corrects], [num_fg, num_bg]


class ExpandLoss(nn.Module):
    def __init__(self, d_fg, d_bg):
        super(ExpandLoss, self).__init__()
        self.d_fg = d_fg
        self.d_bg = d_bg

    def forward(self, predicts):
        seg_preds = torch.softmax(predicts, dim=1)
        fg_seg_preds = seg_preds[:, 1, :, :].view(seg_preds.size(0), -1)
        fg_sorted_preds, _ = torch.sort(fg_seg_preds, dim=1, descending=True)
        fg_weights = torch.tensor([self.d_fg ** i for i in range(fg_sorted_preds.size(1))]).to(seg_preds.device)
        weighted_fg_preds = fg_sorted_preds * fg_weights

        bg_seg_preds = seg_preds[:, 0, :, :].view(seg_preds.size(0), -1)
        bg_sorted_preds, _ = torch.sort(bg_seg_preds, dim=1, descending=True)
        bg_weights = torch.tensor([self.d_bg ** i for i in range(fg_sorted_preds.size(1))]).to(seg_preds.device)
        weighted_bg_preds = bg_sorted_preds * bg_weights

        g_fg = 1/torch.sum(fg_weights) * torch.sum(weighted_fg_preds, dim=1)
        g_bg = 1/torch.sum(bg_weights) * torch.sum(weighted_bg_preds, dim=1)
        loss_fg = -torch.mean(torch.log(g_fg))
        loss_bg = -torch.mean(torch.log(g_bg))
        return loss_fg + loss_bg


class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, *input):
        raise NotImplementedError


class _WeightedLoss(_Loss):
    def __init__(self, weight=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(reduction)
        self.register_buffer('weight', weight)

    def forward(self, *input):
        raise NotImplementedError


class DynamicLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, gamma=0, weight=None, ignore_index=-255, reduction='mean'):
        super().__init__(weight, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

    def forward(self, inputs, targets):
        targets = self._scale_target(targets, (inputs.size(2), inputs.size(3)))
        total_loss = self.criterion_ce(inputs, targets)  # Pixel-level weight is always 1

        if self.gamma == 0:  # No dynamic loss
            total_loss = total_loss.sum() / (targets != self.ignore_index).sum()
        else:  # Dynamic loss
            probabilities = inputs.softmax(dim=1).clone().detach()
            fg_indices = targets.unsqueeze(1).clone().detach()
            bg_indices = 1 - fg_indices
            fg_indices[fg_indices == self.ignore_index] = 0
            bg_indices[bg_indices < 0] = 0
            fg_probabilities = probabilities.gather(dim=1, index=fg_indices).squeeze(1)
            bg_probabilities = probabilities.gather(dim=1, index=bg_indices).squeeze(1)
            fg_probabilities = fg_probabilities ** self.gamma
            bg_probabilities = bg_probabilities ** self.gamma
            probabilities = fg_probabilities + bg_probabilities
            total_loss = (total_loss * probabilities).sum() / (targets != self.ignore_index).sum()
        return total_loss







