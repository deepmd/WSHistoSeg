import os
import csv
from glob import glob
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data_path = "../save/glas_model_resnet50_SGD_lr_0.01_bsz_32_loss_CE_CL_trial_3028"
    dirs_path = glob(os.path.join(data_path, 'C*'))

    for dir_path in dirs_path:
        for csv_name in sorted(os.listdir(dir_path)):
            label = "".join(os.path.splitext(csv_name)[0].split("_"))
            steps, values = [], []
            with open(os.path.join(dir_path, csv_name), 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile)
                for row in csv_reader:
                    steps.append(int(row['Step']))
                    values.append(float(row['Value']))
            # new_steps = np.linspace(min(steps), max(steps), 2000)
            # spl = make_interp_spline(steps, values, k=3)
            # values_smooth = spl(new_steps)
            plt.plot(steps, values, label=label.capitalize())
        plt.legend()
        # Add text annotation
        # plt.text(2, 0.5, 'Example Plot', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.xlabel('Epochs')
        plt.ylabel('Average Number of Correctly Sampled Pixels')
        plt.savefig(f'cofg_cobg_plots/{os.path.basename(dir_path)}.png')
        plt.clf()  # Clear the current figure for the next iteration
