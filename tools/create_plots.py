import os
from os import path
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dir_path = "plots"
    os.makedirs(dir_path, exist_ok=True)
    model_curve_path = path.join(dir_path, 'split_metrics_test_ours.npy')
    negev_curve_path = path.join(dir_path, 'split_metrics_test_negev.npy')
    gradcam_curve_path = path.join(dir_path, 'split_metrics_test_gradcam.npy')

    model_metrics = np.load(model_curve_path, allow_pickle=True)
    negev_metrics = np.load(negev_curve_path, allow_pickle=True)
    gradcam_metrics = np.load(gradcam_curve_path, allow_pickle=True)
    # model_metrics = model_metrics.item().get('mdice').astype(float)
    # negev_metrics = negev_metrics.item().get('mdice').astype(float)
    # gradcam_metrics = gradcam_metrics.item().get('mdice').astype(float)

    # cam_threshold_list = list(np.arange(0, 1, 0.001))
    # threshold_list_right_edge = np.append(cam_threshold_list)
    # threshold_list_right_edge = np.append(cam_threshold_list, [1.0, 2.0, 3.0])
    threshold_list_right_edge = np.arange(0, 1, 0.001)

    model_precision = model_metrics.item().get('precision').astype(float)
    negev_precision = negev_metrics.item().get('precision').astype(float)
    gradcam_precision = gradcam_metrics.item().get('precision').astype(float)

    model_recall = model_metrics.item().get('recall').astype(float)
    negev_recall = negev_metrics.item().get('recall').astype(float)
    gradcam_recall = gradcam_metrics.item().get('recall').astype(float)

    # Create the plot
    plt.figure(figsize=(8, 6))  # Set the figure size
    plt.plot(model_recall, model_precision, label='Ours', color='red')
    plt.plot(negev_recall, negev_precision, label='NEGEV', color='green')
    plt.plot(gradcam_recall, gradcam_precision, label='Grad-CAM', color='blue')

    # Add a legend
    plt.legend(loc='upper right')

    # Add a description box
    # description = "This is a plot with three curves.\nEach curve is displayed in a different color."
    # plt.gcf().text(0.15, 0.1, description, fontsize=10, ha='left')

    # Set labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Precision VS. Recall')

    # Save the plot
    plt.savefig(path.join(dir_path, 'precision-recall-curve.png'))

    # Show the plot
    plt.grid(True)  # Add grid lines
    plt.show()
    plt.pause(10)



