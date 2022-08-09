
from utils import TenFCVDataset, ArgCenter
from utils import MyDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import numpy as np

def train(dataset, subject, fold):
    args = ArgCenter(dataset).get_arg()
    args.eval_idx = fold
    args.eval_subject = subject

    tfcv = TenFCVDataset(subject=subject, args=args, fold=args.eval_idx)
    x_train, y_train, x_val, y_val = tfcv.get_data()

    args.val_len = x_val.shape[0]

    train_loader = MyDataset(x_train, y_train)
    val_loader = MyDataset(x_val, y_val)

    train_iter = DataLoader(dataset=train_loader, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers)
    val_iter = DataLoader(dataset=val_loader, batch_size=args.batch_size*4, shuffle=False,
                          num_workers=args.num_workers)

    trainer = Trainer(args)

    args.train_mode = 'llt'
    trainer.train(train_iter, val_iter)

    args.train_mode = 'hlt'
    trainer.train(train_iter, val_iter)


if __name__ == '__main__':
    train(dataset='BCIC', subject=1, fold=1)  # Key: BCIC, PhysioNet, Cho, Lee
