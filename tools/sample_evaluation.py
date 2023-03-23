import sys
from glob import glob
import os
import numpy as np
import cv2


def get_stats(indices, shape, region='foreground'):
    stats = dict()
    const = 1 if region == 'foreground' else 0
    selected_indices = np.zeros(shape)
    for last_index in range(1, len(indices)+1):
        for first_index in range(last_index):
            selected_indices[indices[first_index]] = 1
        stats[last_index] = (mask[selected_indices == 1] == const).sum()
        selected_indices = np.zeros(shape)
    return stats


if __name__ == '__main__':
    cam_dir = sys.argv[1]
    mask_dir = sys.argv[2]
    num_samples = int(sys.argv[3])
    total_foreground_stats = {i+1: 0 for i in range(num_samples)}
    total_background_stats = {i+1: 0 for i in range(num_samples)}

    cam_paths = glob(os.path.join(cam_dir, '*.npy'))
    for idx, cam_path in enumerate(cam_paths):
        file_name = os.path.basename(cam_path).replace('.npy', '')
        mask_name = file_name.replace('.', '_anno.')
        mask_path = os.path.join(mask_dir, mask_name)

        cam = np.load(cam_path)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0.5).astype(np.uint8)
        mask = cv2.resize(mask, cam.shape, interpolation=cv2.INTER_NEAREST)

        cam = cam.reshape(-1)
        mask = mask.reshape(-1)

        foreground_indices = np.flip(np.argsort(cam)[-num_samples:])
        background_indices = np.argsort(cam)[:num_samples]

        foreground_stats = get_stats(foreground_indices, cam.shape)
        background_stats = get_stats(background_indices, cam.shape, region='background')

        for key, value in foreground_stats.items():
            total_foreground_stats[key] += value
        for key, value in background_stats.items():
            total_background_stats[key] += value

        with open(cam_path + '_stats.txt', 'w') as f:
            f.write('foreground: ' + str(foreground_stats))
            f.write('\n')
            f.write('background: ' + str(background_stats))

        print(f"{idx+1}/{len(cam_paths)}")

    with open(os.path.join(cam_dir, 'total_stats.txt'), 'w') as f:
        f.write('Foreground:\n')
        for key, value in total_foreground_stats.items():
            f.write(f'{key}: {value / (key * 67) * 100:0.2f},\n')
        f.write('\nBackground\n')
        for key, value in total_background_stats.items():
            f.write(f'{key}: {value / (key * 67) * 100:0.2f},\n')





