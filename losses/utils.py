import torch


# def generate_foreground_background_mask(cams, ignore_index, num_samples):
#     batch_size, h, w = cams.size()
#
#     flatten_cams = cams.view(batch_size, -1)
#     _, fg_indices = torch.topk(flatten_cams, num_samples, dim=1, largest=True)
#     _, bg_indices = torch.topk(flatten_cams, num_samples, dim=1, largest=False)
#
#     mask = torch.ones_like(flatten_cams, dtype=torch.uint8) * ignore_index
#     mask.scatter_(1, fg_indices, 1)
#     mask.scatter_(1, bg_indices, 0)
#
#     mask = mask.view(batch_size, h, w)
#     return mask


def generate_foreground_background_mask(cams, ignore_index, sample_ratio):
    flatten_cams = cams.view(cams.size()[0], -1)
    num_pixels = flatten_cams.size(1)

    sorted_values, _ = torch.sort(flatten_cams, dim=1)
    threshold_index = int(sample_ratio * num_pixels)
    fg_thrshold = sorted_values[:, -threshold_index]
    bg_thrshold = sorted_values[:, threshold_index]

    mask = torch.ones_like(cams, dtype=torch.uint8) * ignore_index
    mask[cams > fg_thrshold.unsqueeze(1).unsqueeze(1)] = 1
    mask[cams <= bg_thrshold.unsqueeze(1).unsqueeze(1)] = 0
    return mask


def generate_pseudo_mask_by_cam(cams, ignore_index, sample_ratio):
    flatten_cams = cams.view(cams.size(0), -1)
    sorted_cams, _ = torch.sort(flatten_cams, dim=1)
    thresh_index = int(sample_ratio * sorted_cams.size(1))
    fg_thresh = sorted_cams[:, -thresh_index].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bg_thresh = sorted_cams[:, thresh_index].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    pseudo_mask = torch.ones_like(cams, dtype=torch.long) * ignore_index
    pseudo_mask[cams > fg_thresh] = 1
    pseudo_mask[cams <= bg_thresh] = 0
    return pseudo_mask


def sample_bg(bg_feats, fg_feats, num_samples=10):
    fg_feats = fg_feats.view(-1, num_samples, fg_feats.size(1))
    avg_fg_feats = torch.mean(fg_feats, dim=1).unsqueeze(1)
    bg_feats = bg_feats.view(fg_feats.size(0), -1, fg_feats.size(2))
    similarities = torch.bmm(avg_fg_feats, bg_feats.transpose(1, 2))
    _, bg_indices = torch.topk(similarities.squeeze(1), k=num_samples, dim=1, largest=True)
    selected_bg_features = bg_feats[torch.arange(bg_indices.size(0)).unsqueeze(1), bg_indices]
    selected_bg_features = selected_bg_features.view(-1, selected_bg_features.size(-1))
    return selected_bg_features, bg_indices


def sample_foreground_background_mask(masks, ignore_index, sample_ratio):
    flatten_masks = masks.view(masks.size(0), -1)
    num_pixels = flatten_masks.size(1)
    num_ignored = num_pixels - int(sample_ratio * num_pixels)
    bsz = flatten_masks.size(0)
    mask_ignored_indices = torch.stack([torch.randperm(num_pixels)[:num_ignored] for _ in range(bsz)]).to(masks.device)
    flatten_masks = flatten_masks.scatter(1, mask_ignored_indices, ignore_index)
    return flatten_masks.view(masks.shape)
