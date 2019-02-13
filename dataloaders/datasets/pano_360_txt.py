from __future__ import print_function, division
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Pano_360_txt(Dataset):

    def __init__(self, base_dir, meta_path):
        txt_path = os.path.join(base_dir, meta_path)

        self.base_dir = base_dir
        with open(txt_path) as f:
            self.xy_path = [line.strip().split() for line in f]

    def __len__(self):
        return len(self.xy_path)

    def __getitem__(self, index):
        rgb_path = os.path.join(self.base_dir, self.xy_path[index][0])
        sem_path = os.path.join(self.base_dir, self.xy_path[index][1])
        rgb = np.array(Image.open(rgb_path))[..., :3]
        sem = np.array(Image.open(sem_path)) - 1

        # Random flip
        if np.random.randint(2) == 0:
            rgb = np.flip(rgb, 1)
            sem = np.flip(sem, 1)

        # Random rotation
        shift = np.random.randint(rgb.shape[1])
        rgb = np.roll(rgb, shift, 1)
        sem = np.roll(sem, shift, 1)

        # Convert to tensor
        rgb = torch.from_numpy(rgb.astype(np.float32).transpose((2, 0, 1)))
        sem = torch.from_numpy(sem.astype(np.float32))

        return {
            'image': rgb,
            'label': sem,
        }
