import os
os.chdir('C:/Users/pfbur/Box/projects/CFL-GIP/VaDE_code/Pytorch-VaDE')

import argparse 
import torch.utils.data
from torchvision import datasets, transforms

from train import TrainerVaDE
from preprocess import get_mnist


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2,
                        help="number of iterations")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=100, 
                        help="Batch size")
    parser.add_argument('--pretrain', type=bool, default=False,
                        help='learning rate')
    parser.add_argument('--pretrained_path', type=str, default='weights/pretrained_parameter.pth',
                        help='Output path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = get_mnist(batch_size=args.batch_size)
    
    vade = TrainerVaDE(args, device, dataloader)
    if args.pretrain==True:
        vade.pretrain()
    vade.train()

