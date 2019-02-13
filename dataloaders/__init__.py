from dataloaders.datasets import pano_360_txt
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):

    train_set = pano_360_txt.Pano_360_txt(args.data_base_dir, args.train_meta)
    if args.valid_meta:
        valid_set = pano_360_txt.Pano_360_txt(args.data_base_dir, args.valid_meta)
    else:
        valid_set = None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader
