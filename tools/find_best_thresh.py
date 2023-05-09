from os import path
from glob import glob

import numpy as np

if __name__ == "__main__":
    mask_dir = "/home/reza/Documents/GLAS/Warwick_QU_Dataset_(Released_2016_07_08)"
    cam_dir = path.join(mask_dir, "CAMs/Layer4")
    mask_paths = glob(path.join(mask_dir, '*_anno.bmp'))

    fg_threshs = np.arange(0.4, 1.0, 0.1).tolist() + [1.0]
    bg_threshs = np.arange(0, 0.5, 0.1).tolist()

    # for fg_thresh in fg_threshs:
    #     for bg_thresh in bg_threshs:


