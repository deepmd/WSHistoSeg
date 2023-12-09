import numpy as np
from easydict import EasyDict

# partial names of metrics
constants = EasyDict()
constants.MTR_PXAP = 'PXAP'
constants.MTR_TP = 'True positive'
constants.MTR_FN = 'False negative'
constants.MTR_TN = 'True negative'
constants.MTR_FP = 'False positive'
constants.MTR_DICEFG = 'Dice foreground'
constants.MTR_DICEBG = 'Dice background'
constants.MTR_MIOU = 'MIOU'
constants.MTR_BESTTAU = 'Best tau'
constants.MTR_CL = 'Classification accuracy'


class MaskEvaluation(object):
    def __init__(self, cam_curve_interval, best_valid_tau=None):
        self.best_valid_tau = best_valid_tau

        self.best_tau_list = []
        self.curve_s = None

        self.perf_gist = None

        # cam_threshold_list is given as [0, bw, 2bw, ..., 1-bw]
        # Set bins as [0, bw), [bw, 2bw), ..., [1-bw, 1), [1, 2), [2, 3)
        self.cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
        self.num_bins = len(self.cam_threshold_list) + 2
        self.threshold_list_right_edge = np.append(self.cam_threshold_list,
                                                   [1.0, 2.0, 3.0])

        self.gt_true_score_hist = np.zeros(self.num_bins, dtype=float)
        self.gt_false_score_hist = np.zeros(self.num_bins, dtype=float)

    @staticmethod
    def check_scoremap_validity(scoremap):
        if not isinstance(scoremap, np.ndarray):
            raise TypeError(f"Scoremap must be a numpy array; it is {type(scoremap)}.")
        if scoremap.dtype != float:
            raise TypeError(
                f"Scoremap must be of np.float type; it is of {scoremap.dtype} type."
            )
        if len(scoremap.shape) != 2:
            raise ValueError(f"Scoremap must be a 2D array; it is {len(scoremap.shape)}D.")
        if np.isnan(scoremap).any():
            raise ValueError("Scoremap must not contain nans.")
        if (scoremap > 1).any() or (scoremap < 0).any():
            raise ValueError(
                f"Scoremap must be in range [0, 1].scoremap.min()={scoremap.min()}, scoremap.max()={scoremap.max()}."
            )

    def accumulate(self, scoremap, gt_mask):
        """
        Score histograms over the score map values at GT positive and negative
        pixels are computed.
        """
        MaskEvaluation.check_scoremap_validity(scoremap)

        gt_true_scores = scoremap[gt_mask == 1]
        gt_false_scores = scoremap[gt_mask == 0]

        # histograms in ascending order
        gt_true_hist, _ = np.histogram(gt_true_scores,
                                       bins=self.threshold_list_right_edge)
        self.gt_true_score_hist += gt_true_hist.astype(float)

        gt_false_hist, _ = np.histogram(gt_false_scores,
                                        bins=self.threshold_list_right_edge)
        self.gt_false_score_hist += gt_false_hist.astype(float)

    def get_best_operating_point(self, miou, tau: float = None):
        if tau is None:
            idx = np.argmax(miou)
        else:
            idx = np.argmin(
                np.abs(np.array(self.threshold_list_right_edge) - tau))

        return self.threshold_list_right_edge[idx]

    def compute(self):
        """
        Arrays are arranged in the following convention (bin edges):

        gt_true_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        gt_false_score_hist: [0.0, eps), ..., [1.0, 2.0), [2.0, 3.0)
        tp, fn, tn, fp: >=2.0, >=1.0, ..., >=0.0

        Returns:
            auc: float. The area-under-curve of the precision-recall curve.
               Also known as average precision (AP).
        """
        num_gt_true = self.gt_true_score_hist.sum()
        tp = self.gt_true_score_hist[::-1].cumsum()
        fn = num_gt_true - tp

        num_gt_false = self.gt_false_score_hist.sum()
        fp = self.gt_false_score_hist[::-1].cumsum()
        tn = num_gt_false - fp

        if ((tp + fn) <= 0).all():
            raise RuntimeError("No positive ground truth in the eval set.")
        if ((tp + fp) <= 0).all():
            raise RuntimeError("No positive prediction in the eval set.")

        non_zero_indices = (tp + fp) != 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        auc = (precision[1:] * np.diff(recall))[non_zero_indices[1:]].sum()
        auc *= 100

        dice_fg = 2. * tp / (2. * tp + fp + fn)
        dice_bg = 2. * tn / (2. * tn + fp + fn)
        mdice = 0.5 * (dice_fg + dice_bg)

        iou_fg = tp / (tp + fp + fn)
        iou_bg = tn / (tn + fp + fn)
        miou = 0.5 * (iou_fg + iou_bg)
        # miou = np.array(list(map(lambda x: round(x, 3), miou)))
        # miou = np.array([miou[index] for index in np.arange(0.0, 1.0, 0.1)])

        if self.best_valid_tau is None:
            self.best_tau_list = [self.get_best_operating_point(
                miou=miou, tau=None)]
        else:
            self.best_tau_list = [self.best_valid_tau]

        idx = np.argmin(np.abs(
            self.threshold_list_right_edge - self.best_tau_list[0]))

        total_fg = float(tp[idx] + fn[idx])
        total_bg = float(tn[idx] + fp[idx])

        self.perf_gist = {
            constants.MTR_PXAP: auc,
            constants.MTR_TP: 100 * tp[idx] / total_fg,
            constants.MTR_FN: 100 * fn[idx] / total_fg,
            constants.MTR_TN: 100 * tn[idx] / total_bg,
            constants.MTR_FP: 100 * fp[idx] / total_bg,
            constants.MTR_DICEFG: 100 * dice_fg[idx],
            constants.MTR_DICEBG: 100 * dice_bg[idx],
            constants.MTR_MIOU: 100 * miou[idx],
            constants.MTR_BESTTAU: self.best_tau_list,
            'mdice_best': 100 * mdice[idx],
            'miou': miou,
            'mdice': mdice,
            'dice_fg': dice_fg,
            'dice_bg': dice_bg,
            'precision': precision,
            'recall': recall
        }

        self.curve_s = {
            'x': recall,
            'y': precision,
            constants.MTR_MIOU: 100. * miou,
            constants.MTR_TP: 100. * tp / total_fg,
            constants.MTR_TN: 100. * tn / total_bg,
            constants.MTR_FP: 100. * fp / total_bg,
            constants.MTR_FN: 100. * fn / total_fg,
            constants.MTR_DICEFG: dice_fg,
            constants.MTR_DICEBG: dice_bg,
            constants.MTR_BESTTAU: self.best_tau_list,
            'idx': idx,
            'num_bins': self.num_bins,
            'threshold_list_right_edge': self.threshold_list_right_edge,
            'cam_threshold_list': self.cam_threshold_list,
            'perf_gist': self.perf_gist
        }

        return auc








