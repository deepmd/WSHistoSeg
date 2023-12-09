import os


def get_image_ids(root, suffix=None):
    """
    image_ids.txt has the structure

    <path>
    path/to/image1.jpg
    path/to/image2.jpg
    path/to/image3.jpg
    ...
    """
    file_path = os.path.join(root, f'image_ids_{suffix}.txt' if suffix else 'image_ids.txt')
    with open(file_path) as file:
        return [line.strip() for line in file]


def get_class_labels(root):
        """
        class_labels.txt has the structure

        <path>,<integer_class_label>
        path/to/image1.jpg,0
        path/to/image2.jpg,1
        path/to/image3.jpg,1
        ...
        """
        file_path = os.path.join(root, 'class_labels.txt')
        class_labels = {}
        with open(file_path) as file:
            for line in file:
                image_id, class_label_string = line.strip().split(',')
                class_labels[image_id] = int(class_label_string)
        return class_labels


def get_mask_paths(root):
    """
        localization.txt (for masks) has the structure

        <path>,<link_to_mask_file>,<link_to_ignore_mask_file>
        path/to/image1.jpg,path/to/mask1a.png,path/to/ignore1.png
        path/to/image1.jpg,path/to/mask1b.png,
        path/to/image2.jpg,path/to/mask2a.png,path/to/ignore2.png
        path/to/image3.jpg,path/to/mask3a.png,path/to/ignore3.png
        ...

        One image may contain multiple masks (multiple mask paths for same image).
        One image contains only one ignore mask.
        """
    file_path = os.path.join(root, 'localization.txt')
    mask_paths = {}
    ignore_paths = {}
    with open(file_path) as file:
        for line in file:
            image_id, mask_path, ignore_path = line.strip('\n').split(',')
            if image_id in mask_paths:
                mask_paths[image_id].append(mask_path)
                assert (len(ignore_path) == 0)
            else:
                mask_paths[image_id] = [mask_path]
                ignore_paths[image_id] = ignore_path
    return mask_paths, ignore_paths


def get_cam_paths(root):
    file_path = os.path.join(root, 'image_cams.txt')
    cam_paths = {}
    with open(file_path) as file:
        for line in file:
            line_parts = line.strip().split(',')
            image_id, cam_file_paths = line_parts[0], line_parts[1]
            cam_paths[image_id] = cam_file_paths
    return cam_paths
