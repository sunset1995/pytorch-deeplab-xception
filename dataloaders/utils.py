import matplotlib.pyplot as plt
import numpy as np
import torch


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]).astype(np.float32))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'stanford2d3d':
        label_colours = get_stanford2d3d_labels()
    elif dataset == 'mpv3':
        label_colours = get_mpv3_labels()
    elif dataset == 'sumo':
        label_colours = get_sumo_labels()
    elif dataset == 'random_room_v5':
        label_colours = get_random_room_v5_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, len(label_colours)):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def get_mpv3_labels():
    return np.array([
        [190, 153, 153],
        [107, 142, 35],
        [220, 20, 60],
        [255, 0, 0],
        [153, 153, 153],
        [128, 64, 128],
        [152, 251, 152],
        [0, 130, 180],
        [244, 35, 232],
        [220, 220, 0],
        [250, 170, 30],
        [70, 70, 70],
        [102, 102, 156]])


def get_random_room_v5_labels():
    return np.array([
        [128, 64, 128],
        [250, 170, 30],
        [153, 153, 153],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [107, 142, 35],
        [190, 153, 153]])


def get_stanford2d3d_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0]])


def get_sumo_labels():
    return np.array([
        [122, 197, 205], [0, 178, 238], [176, 226, 255], [83, 134, 139], [176, 196, 222],
        [72, 61, 139], [30, 144, 255], [16, 78, 139], [0, 0, 205], [132, 112, 255],
        [188, 210, 238], [175, 238, 238], [65, 105, 225], [0, 34, 102], [39, 64, 139],
        [106, 90, 205], [122, 103, 238], [24, 116, 205], [74, 112, 139], [127, 255, 212],
        [139, 139, 0], [47, 79, 79], [79, 148, 205], [238, 180, 180], [165, 42, 42],
        [95, 158, 160], [100, 149, 237], [199, 21, 133], [102, 139, 139], [28, 134, 238],
        [0, 0, 139], [0, 238, 238], [131, 111, 255], [0, 134, 139], [0, 100, 0],
        [155, 205, 155], [173, 216, 230], [0, 255, 255], [179, 238, 58], [84, 255, 159],
        [0, 205, 102], [164, 211, 238], [151, 255, 255], [0, 255, 0], [205, 155, 155],
        [139, 35, 35], [210, 180, 140], [102, 205, 170], [25, 25, 112], [205, 170, 125],
        [255, 165, 79], [255, 140, 0], [154, 205, 50], [205, 133, 63], [139, 69, 0],
        [174, 238, 238], [126, 192, 238], [255, 218, 185], [255, 114, 86], [118, 238, 198],
        [0, 0, 238], [0, 205, 205], [0, 229, 238], [188, 143, 143], [139, 69, 19],
        [255, 246, 143], [238, 118, 33], [189, 183, 107], [173, 255, 47], [152, 251, 152],
        [143, 188, 143], [69, 139, 0], [32, 178, 170], [0, 139, 0], [205, 112, 84],
        [255, 0, 0], [144, 238, 144], [255, 110, 180], [0, 238, 118], [205, 198, 115],
        [255, 127, 0], [127, 255, 0], [0, 205, 0], [255, 160, 122], [238, 118, 0],
        [255, 69, 0], [255, 218, 185], [233, 150, 122], [238, 203, 173], [255, 228, 196],
        [255, 127, 80], [139, 90, 0], [205, 104, 57], [139, 58, 98], [255, 20, 147],
        [238, 99, 99], [205, 55, 0], [255, 130, 171], [255, 165, 0], [178, 34, 34],
        [238, 0, 0], [139, 58, 58], [255, 255, 0], [139, 0, 0], [178, 58, 238], [238, 64, 0],
        [139, 34, 82], [238, 58, 140], [176, 48, 96], [205, 38, 38], [238, 92, 66],
        [104, 34, 139], [205, 0, 0], [205, 105, 201], [139, 102, 139], [205, 79, 57],
        [122, 55, 139], [159, 121, 238], [85, 26, 139], [191, 62, 255], [205, 41, 144],
        [218, 112, 214], [171, 130, 255], [186, 85, 211], [137, 104, 205], [245, 222, 179],
        [221, 160, 221], [145, 44, 238], [139, 0, 139], [139, 105, 20], [205, 186, 150],
        [139, 101, 8]
    ])


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])


def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])
