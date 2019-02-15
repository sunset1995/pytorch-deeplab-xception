import argparse
import os
import numpy as np
from tqdm import tqdm

from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define network
        model = DeepLab(num_classes=args.num_classes,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        self.model = model
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        _, self.valid_loader = make_data_loader(args, **kwargs)
        self.pred_remap = args.pred_remap
        self.gt_remap = args.gt_remap

        # Define Evaluator
        self.evaluator = Evaluator(args.eval_num_classes)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def validation(self):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.valid_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)

            if self.gt_remap is not None:
                target = self.gt_remap[target.astype(int)]
            if self.pred_remap is not None:
                pred = self.pred_remap[pred.astype(int)]
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--data-base-dir', type=str, default='/home/sunset/Dataset360/')
    parser.add_argument('--valid-meta', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['stanford2d3d', 'mpv3',
                                 'sumo', 'random_room_v5'])
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--eval-num-classes', type=int, required=True)
    parser.add_argument('--resume', type=str, required=True,
                        help='put the path to resuming file if needed')
    parser.add_argument('--pred-remap', type=str, default=None)
    parser.add_argument('--gt-remap', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=12,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--test-batch-size', type=int, default=12)
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluuation interval (default: 1)')

    args = parser.parse_args()
    args.train_meta = None
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    if args.pred_remap is not None:
        remap = np.zeros(256, np.int32) + 255
        with open(args.pred_remap) as f:
            for si, ti in enumerate(f):
                ti = int(ti.split()[0])
                remap[si] = ti
        args.pred_remap = remap

    if args.gt_remap is not None:
        remap = np.zeros(256, np.int32) + 255
        with open(args.gt_remap) as f:
            for si, ti in enumerate(f):
                ti = int(ti.split()[0])
                remap[si] = ti
        args.gt_remap = remap

    trainer = Trainer(args)
    trainer.validation()


if __name__ == "__main__":
    main()
