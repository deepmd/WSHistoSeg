
from os import path, makedirs
from glob import glob
from matplotlib import cm

import numpy as np
from PIL import Image
import cv2
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image


# def overlay_mask(image: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
#     cmap = cm.get_cmap(colormap)
#
#     mask = mask.resize(image.size, resample=Image.BICUBIC)
#     overlay = (255 * cmap(np.asarray(mask).astype(np.float) ** 2)[:, :, :3]).astype(np.uint8)
#     overlay_img = Image.fromarray((alpha * np.asarray(image) + (1 - alpha) * overlay).astype(np.uint8))
#     return overlay_img


def save_image(image: Image.Image, image_path: str):
    image.save(image_path)


if __name__ == "__main__":
    image_dir = "/home/reza/Documents/GLAS/Warwick_QU_Dataset_(Released_2016_07_08)"
    cam_dir = "/home/reza/Documents/GLAS/Warwick_QU_Dataset_(Released_2016_07_08)/CAMs/Layer4"
    output_dir = "./overlay_images4"
    makedirs(output_dir, exist_ok=True)

    file_paths = glob(path.join(image_dir, 'test*.bmp'))
    image_paths = [file_name for file_name in file_paths if 'anno' not in file_name]
    mask_paths = [image_name.replace('.', '_anno.') for image_name in image_paths]
    cam_paths = [path.join(cam_dir, path.basename(image_name).replace('.bmp', '_layer4_cam.npy'))
                 for image_name in image_paths]
    # cam_paths = [path.join(cam_dir, path.basename(image_name).replace('.bmp', '_negev.npy')) for image_name in image_paths]

    for counter, (image_path, mask_path, cam_path) in enumerate(zip(image_paths, mask_paths, cam_paths)):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)

        # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # mask = (mask > 0.5).astype(np.float32)
        #
        # overlay_image = overlay_mask(to_pil_image(image), to_pil_image(mask))
        # image_name = path.basename(image_path).replace('.bmp', '.png')
        # output_path = path.join(output_dir, image_name)
        # save_image(overlay_image[0], output_path)

        cam = np.load(cam_path).astype(np.float32)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        overlay_image = overlay_mask(to_pil_image(image), to_pil_image(cam))
        cam_name = path.basename(image_path).replace('.bmp', '_gradcam.png')
        output_path = path.join(output_dir, cam_name)
        save_image(overlay_image[0], output_path)


        # image = Image.open(image_path)
        # mask = Image.open(mask_path).convert("L")
        #
        # # overlay_image = overlay_mask(image, mask)
        # # image_name = path.basename(image_path).replace('.bmp', '.png')
        # # output_path = path.join(output_dir, image_name)
        # # save_image(overlay_image, output_path)
        #
        # cam = Image.fromarray(np.load(cam_path))
        # overlay_image = overlay_mask(image, cam)
        # cam_name = path.basename(image_path).replace('.bmp', '_our.png')
        # output_path = path.join(output_dir, cam_name)
        # save_image(overlay_image, output_path)



