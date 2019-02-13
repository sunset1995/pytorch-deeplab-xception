import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        rgb = image[:3].clone().cpu().data / 255
        y_ = decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(), dataset=dataset)
        y = decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(), dataset=dataset)
        rgb_y_ = (rgb + y_) / 2
        rgb_y = (rgb + y) / 2

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
