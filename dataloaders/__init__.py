from dataloaders.datasets import pano_360_txt
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    if args.train_meta:
        train_set = pano_360_txt.Pano_360_txt(args.data_base_dir, args.train_meta, augmentation=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        train_loader = None

    if args.valid_meta:
        valid_set = pano_360_txt.Pano_360_txt(args.data_base_dir, args.valid_meta, augmentation=False)
        valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        valid_loader = None

    return train_loader, valid_loader
