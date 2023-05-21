import torch
from torch import nn
import torch.nn.functional as F
from .utils import generate_foreground_background_mask, generate_pseudo_mask_by_cam, sample_bg


# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, class_weights, ignore_index):
        super(FSCELoss, self).__init__()
        weight = torch.FloatTensor(class_weights)
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='mean')

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.ce_loss(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.ce_loss(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().float()
        targets = F.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class ContrastCELoss(nn.Module):
    def __init__(self, class_weights, ignore_index, loss_weight,
                 temperature, base_temperature, num_samples):
        super(ContrastCELoss, self).__init__()

        self.loss_weight = loss_weight
        self.ignore_index = ignore_index
        self.seg_criterion = FSCELoss(class_weights=class_weights, ignore_index=self.ignore_index)
        self.contrast_criterion = PixelContrastLoss(temperature=temperature, base_temperature=base_temperature,
                                                    ignore_index=self.ignore_index, num_samples=num_samples)

    def forward(self, preds, cams, masks, labels, use_pseudo_mask=False):
        assert "seg" in preds
        assert "embed" in preds

        seg_preds = preds['seg']
        embedding = preds['embed']

        target_mask = generate_pseudo_mask_by_cam(cams, self.ignore_index) if use_pseudo_mask else masks
        target_mask = target_mask.to(seg_preds.device)

        loss = self.seg_criterion(seg_preds, target_mask)

        loss_contrast, num_corrects = self.contrast_criterion(embedding, cams, masks, labels)

        return loss + self.loss_weight * loss_contrast, {'ce': loss, 'contrast': loss_contrast}, num_corrects


class PixelContrastLoss(nn.Module):
    def __init__(self, temperature, base_temperature, ignore_index, num_samples):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_samples = num_samples
        self.ignore_index = ignore_index

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

    def forward(self, embeddings, cams, masks, labels=None):
        h, w = embeddings.size()[2:]
        masks = F.interpolate(masks.float(), (h, w), mode='nearest')
        masks = masks.squeeze(1).long()

        cams = F.interpolate(cams, (h, w), mode='bicubic', align_corners=True)
        cams = cams.squeeze(1)

        embeddings = embeddings.permute(0, 2, 3, 1)

        selected_pixels = generate_foreground_background_mask(cams, self.num_samples)

        fg_feats = embeddings[selected_pixels == 1]
        fg_labels = torch.ones(fg_feats.shape[0])

        bg_feats = embeddings[selected_pixels == 0]
        # bg_feats = sample_bg(bg_feats, fg_feats)
        bg_labels = torch.zeros(bg_feats.shape[0])

        all_feats = torch.cat([fg_feats, bg_feats], dim=0).unsqueeze(1)
        all_labels = torch.cat([fg_labels, bg_labels], dim=0)

        contrast_loss = self._contrastive(all_feats, all_labels)

        num_fg_corrects = int((masks[selected_pixels == 1] == 1).sum())
        num_bg_corrects = int((masks[selected_pixels == 0] == 0).sum())
        return contrast_loss, [num_fg_corrects, num_bg_corrects]
