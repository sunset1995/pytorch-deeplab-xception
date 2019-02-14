import os
import torch
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory
        self.mean = torch.FloatTensor(
            np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        ).cuda()
        self.std = torch.FloatTensor(
            np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        ).cuda()

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        rgb = (image[:3] * self.std + self.mean).clone().cpu().data
        y_ = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(), dataset=dataset)
        y = decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(), dataset=dataset)
        rgb_y_ = rgb * 0.2 + y_ * 0.8
        rgb_y = rgb * 0.2 + y * 0.8

        grid_image = make_grid(rgb, 3, normalize=False, range=(0, 255))
        writer.add_image('Image', grid_image, global_step)

        grid_image = make_grid(y_, 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)

        grid_image = make_grid(y, 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)

        grid_image = make_grid(rgb_y_, 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label + RGB', grid_image, global_step)

        grid_image = make_grid(rgb_y, 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label + RGB', grid_image, global_step)
