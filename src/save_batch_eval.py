from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from supervised_tree import *
from sparse_supervised_tree import *
import argparse


def main():
    
    parser = argparse.ArgumentParser(description="evaluate Supervised Tree-Wasserstein")
    parser.add_argument('--train_path', type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpu", default="cuda", type=str)
    parser.add_argument("--n_inner", default=None, type=int)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    
    train_data = pd.read_csv(args.train_path)
    test_data = pd.read_csv(args.test_path)
    device = args.gpu
    
    y_train = train_data['data_label'].values
    x_train = train_data.drop(['data_label'], axis=1).values 
    y_test = test_data['data_label'].values
    x_test = test_data.drop(['data_label'], axis=1).values 
    
    x_train = torch.tensor(x_train).float()
    y_train = torch.tensor(y_train)
    x_test = torch.tensor(x_test).float()
    y_test = torch.tensor(y_test)
    
    n_train_doc = x_train.shape[0]
    n_test_doc = x_test.shape[0]
    n_leaf = x_train.shape[1]

    if args.n_inner is None:
        model = SupervisedTree(n_leaf, n_leaf, device)
    else:
        model = SupervisedTree(args.n_inner, n_leaf, device)
    model.load_state_dict(torch.load(args.model_path))
    sparse_model = SparseSupervisedTree(model, device)
    sparse_model.to(device)
    
    train_dataset = MyDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=500, shuffle=False)

    test_dataset = MyDataset(x_test, y_test)

    sparse_model.eval()

    n_test_doc = len(test_dataset)

    sorted_label = []
    correct_label = []
    
    for (test_mass, test_label) in tqdm(test_dataset):
    
        test_mass = test_mass.to(device)
        distance_list = []
        label_list = []
        
        for _, (train_mass, train_label) in enumerate(train_loader):

            train_mass = train_mass.to(device)
            distance = sparse_model.calc_distance2(test_mass, train_mass)
            
            distance_list += distance.cpu().tolist()
            label_list += train_label.cpu().tolist()

        sorted_zip = sorted(zip(distance_list, label_list))
        sorted_label.append([item[1] for item in sorted_zip])
        correct_label.append(test_label.item())


    result = (correct_label, sorted_label)
    with open(args.save_path, "wb") as f:
        pickle.dump(result, f)

if __name__ == '__main__':
    main()
