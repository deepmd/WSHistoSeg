import torch
from torch import nn
import torch.nn.functional as F


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
        self.seg_criterion = FSCELoss(class_weights=class_weights, ignore_index=ignore_index)
        self.contrast_criterion = PixelContrastLoss(temperature=temperature, base_temperature=base_temperature,
                                                    num_samples=num_samples)

    def forward(self, preds, target, cams, with_embed=False):
        # h, w = target.size(1), target.size(2)

        assert "seg" in preds
        assert "embed" in preds

        pred = seg = preds['seg']
        embedding = preds['embed']

        # target = F.interpolate(target.float(), pred.size()[2:], mode='nearest')
        # pred = F.interpolate(input=seg, size=(h, w), mode='bilinear', align_corners=True)
        loss = self.seg_criterion(pred, target)

        loss_contrast = 0.0
        if with_embed is True:
            # predict = torch.argmax(seg, 1)
            loss_contrast = self.contrast_criterion(embedding, target, cams)

            return loss + self.loss_weight * loss_contrast, {'ce': loss, 'contrast': loss_contrast}

        # just a trick to avoid errors in distributed training
        return loss + 0.0 * loss_contrast, {'ce': loss, 'contrast': 0.0}


class PixelContrastLoss(nn.Module):
    def __init__(self, temperature, base_temperature, num_samples):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature
        self.num_samples = num_samples

    # def _hard_anchor_sampling(self, X, y_hat, y):
    #     batch_size, feat_dim = X.shape[0], X.shape[-1]
    #
    #     classes = []
    #     total_classes = 0
    #     for ii in range(batch_size):
    #         this_y = y_hat[ii]
    #         this_classes = torch.unique(this_y)
    #         this_classes = [x for x in this_classes if x != self.ignore_label]
    #         this_classes = [x for x in this_classes if torch.count_nonzero((this_y == x)) > self.max_views]
    #
    #         classes.append(this_classes)
    #         total_classes += len(this_classes)
    #
    #     if total_classes == 0:
    #         return None, None
    #
    #     n_view = self.max_samples // total_classes
    #     n_view = min(n_view, self.max_views)
    #
    #     X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
    #     y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
    #
    #     X_ptr = 0
    #     for ii in range(batch_size):
    #         this_y_hat = y_hat[ii]
    #         this_y = y[ii]
    #         this_classes = classes[ii]
    #
    #         for cls_id in this_classes:
    #             hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero(as_tuple=False)
    #             easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero(as_tuple=False)
    #
    #             num_hard = hard_indices.shape[0]
    #             num_easy = easy_indices.shape[0]
    #
    #             if num_hard >= n_view / 2 and num_easy >= n_view / 2:
    #                 num_hard_keep = n_view // 2
    #                 num_easy_keep = n_view - num_hard_keep
    #             elif num_hard >= n_view / 2:
    #                 num_easy_keep = num_easy
    #                 num_hard_keep = n_view - num_easy_keep
    #             elif num_easy >= n_view / 2:
    #                 num_hard_keep = num_hard
    #                 num_easy_keep = n_view - num_hard_keep
    #             else:
    #                 # Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
    #                 raise Exception
    #
    #             perm = torch.randperm(num_hard)
    #             hard_indices = hard_indices[perm[:num_hard_keep]]
    #             perm = torch.randperm(num_easy)
    #             easy_indices = easy_indices[perm[:num_easy_keep]]
    #             indices = torch.cat((hard_indices, easy_indices), dim=0)
    #
    #             X_[X_ptr, :, :] = X[ii, indices, :].squeeze(1)
    #             y_[X_ptr] = cls_id
    #             X_ptr += 1
    #
    #     return X_, y_

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

    @staticmethod
    def pixel_sampling(cams, num_sample):
        batch_size, h, w = cams.size()
        cams = cams.view(cams.size()[0], -1)
        selected_indices = torch.ones_like(cams, dtype=torch.int) * -255
        sorted_indices = torch.argsort(cams, dim=1, descending=True)
        foreground_indices = sorted_indices[:, :num_sample]
        background_indices = sorted_indices[:, -num_sample:]

        for idx in range(batch_size):
            selected_indices[idx].index_fill_(0, foreground_indices[idx], 1)
            selected_indices[idx].index_fill_(0, background_indices[idx], 0)
        selected_indices = selected_indices.reshape([batch_size, h, w])
        return selected_indices

    def forward(self, feat_embeddings, masks=None, cams=None):
        # masks = F.interpolate(masks.float(), feats.size()[2:], mode='nearest')
        # masks = masks.squeeze(1).long()

        contrast_loss = 0.0
        if not isinstance(feat_embeddings, dict):
            feat_embeddings = {"layer4": feat_embeddings}

        for layer, feats in feat_embeddings.items():
            cam = F.interpolate(cams[layer], feats.size()[2:], mode='bicubic', align_corners=True)
            cam = cam.squeeze(1)

            feats = feats.permute(0, 2, 3, 1)

            selected_pixels = PixelContrastLoss.pixel_sampling(cam, self.num_samples)

            indices = (selected_pixels == 1).nonzero(as_tuple=False)
            fg_feats = [feats[index[0], index[1], index[2], :] for index in indices]
            fg_feats = torch.stack(fg_feats, dim=0)
            fg_labels = torch.ones(fg_feats.shape[0])

            indices = (selected_pixels == 0).nonzero(as_tuple=False)
            bg_feats = [feats[index[0], index[1], index[2], :] for index in indices]
            bg_feats = torch.stack(bg_feats, dim=0)
            bg_labels = torch.zeros(bg_feats.shape[0])

            all_feats = torch.cat([fg_feats, bg_feats], dim=0).unsqueeze(1)
            all_labels = torch.cat([fg_labels, bg_labels], dim=0)

            contrast_loss += self._contrastive(all_feats, all_labels)
        contrast_loss /= len(feat_embeddings.keys())
        return contrast_loss
