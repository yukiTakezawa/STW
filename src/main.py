from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
from supervised_tree import *
import argparse 


def main():

    parser = argparse.ArgumentParser(description="training Supervised Tree-Wasserstein")
    parser.add_argument('--data_path', type=str)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--gpu", default="cuda", type=str)
    parser.add_argument("--lr", default=0.1, type=float)    
    parser.add_argument("--batch", default=30, type=int)
    parser.add_argument("--n_inner", default=None, type=int)
    parser.add_argument("--margin", default=10.0, type=float)
    parser.add_argument("--multiply", default=5.0, type=float)
    args = parser.parse_args()

    train_data = pd.read_csv(args.data_path)
    y_train = train_data['data_label'].values
    x_train = train_data.drop(['data_label'], axis=1).values
    keys = train_data.drop(["data_label"], axis=1).keys()
    x_train = args.multiply * torch.tensor(x_train).float()
    y_train = torch.tensor(y_train)
    n_train_doc = x_train.shape[0]
    n_leaf = x_train.shape[1]
    train_dataset = MyDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    
    if args.eval_path is not None:
        eval_data = pd.read_csv(args.eval_path)
        y_valid = eval_data["data_label"].values
        x_valid = eval_data.drop(["data_label"], axis=1).values
        x_valid = args.multiply * torch.tensor(x_valid).float()
        y_valid = torch.tensor(y_valid)
        n_valid_doc = x_valid.shape[0]
        valid_dataset = MyDataset(x_valid, y_valid)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    device = args.gpu
    
    if args.n_inner is None:
        model = SupervisedTree(n_leaf, n_leaf, device)
    else:
        model = SupervisedTree(args.n_inner, n_leaf, device)
    model.to(device)    
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.eval_path is not None:
        model.run_train(train_loader, args.epoch, optimizer, args.save_path, margin=args.margin, valid_loader=valid_loader)
    else:
        model.run_train(train_loader, args.epoch, optimizer, args.save_path, margin=args.margin)
        
    torch.save(model.state_dict(), args.save_path + "/leatest.pth")

    
if __name__ == '__main__':
    main()
