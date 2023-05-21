import torch


def generate_foreground_background_mask(cams, num_samples):
    batch_size, h, w = cams.size()

    flatten_cams = cams.view(batch_size, -1)
    _, fg_indices = torch.topk(flatten_cams, num_samples, dim=1, largest=True)
    _, bg_indices = torch.topk(flatten_cams, num_samples, dim=1, largest=False)

    mask = torch.ones_like(flatten_cams, dtype=torch.int) * -1
    mask.scatter_(1, fg_indices, 1)
    mask.scatter_(1, bg_indices, 0)

    mask = mask.view(batch_size, h, w)
    return mask

# def generate_foreground_background_mask(cams, percentage):
#     percentage = 0.1
#     flatten_cams = cams.view(cams.size()[0], -1)
#     num_pixels = flatten_cams.size(1)
#
#     sorted_values, _ = torch.sort(flatten_cams, dim=1)
#     threshold_index = int(percentage * num_pixels)
#     fg_thrshold = sorted_values[:, -threshold_index]
#     bg_thrshold = sorted_values[:, threshold_index]
#
#     mask = torch.ones_like(cams, dtype=torch.int) * -1
#     mask[cams > fg_thrshold.unsqueeze(1).unsqueeze(1)] = 1
#     mask[cams <= bg_thrshold.unsqueeze(1).unsqueeze(1)] = 0
#     return mask


def generate_pseudo_mask_by_cam(cams, ignore_index):
        flatten_cams = cams.view(cams.size(0), -1)
        sorted_cams, _ = torch.sort(flatten_cams, dim=1)
        thresh_index = int(0.2 * sorted_cams.size(1))
        fg_thresh = sorted_cams[:, -thresh_index].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        bg_thresh = sorted_cams[:, thresh_index].unsqueeze(1).unsqueeze(1).unsqueeze(1)
        pseudo_mask = torch.ones_like(cams, dtype=torch.long) * ignore_index
        pseudo_mask[cams > fg_thresh] = 1
        pseudo_mask[cams <= bg_thresh] = 0
        return pseudo_mask


def sample_bg(bg_feats, fg_feats):
    avg_fg_feats = torch.mean(fg_feats, dim=0).unsqueeze(0)
    dictances = torch.matmul(bg_feats, torch.transpose(avg_fg_feats, 0, 1))
    _, bg_indices = torch.topk(dictances, k=fg_feats.size(0), dim=0, largest=False)
    return bg_feats[bg_indices].squeeze(1)
