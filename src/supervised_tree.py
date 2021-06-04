import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from math import *
import matplotlib.pyplot as plt
import torch.optim as optim
import random
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pickle


def smoothabs(x, alpha):
    """
    differential absolute function using smooth max.

    Caution.
    if x is large (over 100), this returns nan.

    Parameters
    ----------
    x : torch.tensor
    
    Returns
    ----------
    result : torch.tensor

    """
    return  (x*torch.exp(alpha*x) - x*torch.exp(-alpha*x)) / (2.0 + torch.exp(alpha*x) + torch.exp(-alpha*x))


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    
class SupervisedTree(torch.nn.Module):

    def __init__(self, n_inner, n_leaf, device='cpu', max_d=None, alpha=2.0):
        super(SupervisedTree, self).__init__()
        self.n_inner = n_inner # number of inner nodes which contained root.
        self.n_leaf = n_leaf # number of leaf nodes. (e.g., number of words)
        self.n_node = self.n_leaf + self.n_inner # number of all nodes.
        self.device = device
        self.alpha = alpha
               
        self.param = torch.nn.Parameter(torch.randn(self.n_inner, self.n_leaf, device=self.device)) # D2

        # initialize parameters
        nn.init.normal_(self.param, 0.0, 0.1)
        
        self.A = self.gen_A()
        self.inv_A = self.calc_inv_A() # (I - D1)^{-1}

    def gen_A(self):
        """
        Initialize D1, which is an adjacency matrix of a tree consisting of internal nodes and return I - D1.
        Assume D1 is the perfect 5-ary tree.
        """
        
        A = torch.zeros(self.n_inner, self.n_inner-1)
        for i in range(1, int(self.n_inner/5)):
            A[i-1, 5*(i-1):5*i] = 1.0
        A[int(self.n_inner/5)-1, 5*(int(self.n_inner/5)-1):] = 1.0
        A = A.to(self.device)
        return torch.eye(self.n_inner, device=self.device) - torch.cat([torch.zeros(self.n_inner, 1, device=self.device), A], dim=1).to(self.device)

    
    def calc_inv_A(self):
        """
        return (I - D1)^{-1}.
        """
        return self.A.inverse()

    
    def calc_ppar(self):
        """
        return upper two blocks of D_par.
        """
        
        exp_param = F.softmax(self.param, dim=0)
        return torch.cat([torch.eye(self.n_inner, device=self.device) - self.A, exp_param], dim=1)

    
    def calc_psub(self, block_D=None):
        """
        Retuens
        ----------
        X : torch.tensor (shape is (self.n_inner, self.n_leaf))
            X[i][j] is P_sub (v_j+self.n_inner | v_i).
        """
        
        B = F.softmax(self.param, dim=0)
        X = torch.mm(self.inv_A, B)
        return X

    
    def calc_distance(self, mass1, mass2, block_psub=None):
        """
        Parameters
        ----------
        mass1 : torch.tensor (shape is (self.n_leaf))
            normalized bag-of-words.
        mass2 : torch.tensor (shape is (self.n_leaf))
            normalized bag-of-words
        block_psub : torch.tensor (shape is (self.n_inner, self.n_leaf))
            retuen value of self.calc_plock_psub().
        """
        if block_psub is None:
            block_psub = self.calc_psub()
        return smoothabs(torch.mv(block_psub, mass1 - mass2), alpha=self.alpha).sum() + smoothabs(mass1 - mass2, alpha=self.alpha).sum()

    
    def calc_distances(self, mass, block_psub=None):
        """
        Parameters
        ----------
        mass : torch.tensor (shape is (n_doc, self.n_leaf))
            normalized bag-of-words.
        
        Returns
        ----------
        distances : torch.tensor (shape is (n_doc, n_doc))
            distance[i][j] is tree-wasserstein distance between mass[i] and mass[j].
        """

        if block_psub is None:
            block_psub = self.calc_psub()

        distances = torch.zeros((mass.shape[0], mass.shape[0]), device=self.device)
        for i in range(mass.shape[0]):
            for j in range(i):
                distances[i][j] = self.calc_distance(mass[i], mass[j], block_psub=block_psub)

        return distances + distances.permute(1,0)    

    
    def calc_contrastive_loss(self, distances, labels, margin=10.0):
        """
        calculate contrastive loss
        
        Parameters
        ----------
        distances : torch.tensor (shape is (n_doc, n_doc)):
           return value of self.calc_distances()
        labels torch.tensor (shape is n_doc):
           label of documents.
        margin float:
           margin of contrastive loss.
        
        Returns
        ----------
        loss : float:
            contrastive loss
        """
        n_data = distances.shape[0]
        pos_mask = ((labels.unsqueeze(0) == labels.unsqueeze(1)).float() - torch.eye(n_data, device=self.device))
        neg_mask = 1.0 - pos_mask - torch.eye(n_data, device=self.device)
        n_pos = pos_mask.sum() + 1e-15
        n_neg = neg_mask.sum() + 1e-15
        return (distances * pos_mask).sum()/n_pos + (torch.clamp(-distances * neg_mask, - margin)).sum()/n_neg

    
    def run_train(self, train_loader, epoch, optimizer, save_path=None, margin=10.0, valid_loader=None):
        """
        start training.

        Parameters
        ----------
        train_loader torch.utils.data.DataLoader:
            dataloader for training data
        epoch int:
            number of epochs.
        optimizer:
            optimizer such as Adam.
        save_path str:
            path where model parameter is saved.
            If save_path is not None, then the parameter whose loss for validation data is lowest is saved as best.pth.
        margin float:
            margin of contrastive loss.
        valid_loader torch.utils.data.DataLoader:
            dataloader for validation data.
        """

        loss_list = []
        valid_loss_list = []

        for i_epoch in tqdm(range(epoch)):

            # validation
            if (valid_loader is not None):
                self.eval()
                loss = 0.0
                print("evaluating...")
                for i_batch, (batch_mass, batch_label) in enumerate(valid_loader):
                    batch_mass = batch_mass.to(self.device)
                    batch_label = batch_label.to(self.device)

                    distances = self.calc_distances(batch_mass)
                    loss += self.calc_contrastive_loss(distances, batch_label, margin).item()
                    
                valid_loss_list.append(loss/len(valid_loader))    

                if min(valid_loss_list) == valid_loss_list[-1] and (save_path is not None):
                    torch.save(self.state_dict(), save_path + "/best.pth")
                print(f"valid loss : {loss/len(valid_loader)}")

            #training
            self.train()
            scaler = torch.cuda.amp.GradScaler()
            
            train_loss_list = []
            for i_batch, (batch_mass, batch_label) in enumerate(train_loader):

                batch_mass = batch_mass.to(self.device)
                batch_label = batch_label.to(self.device)
                
                optimizer.zero_grad()            

                with torch.cuda.amp.autocast():
                    distances = self.calc_distances(batch_mass)

                    loss = self.calc_contrastive_loss(distances, batch_label, margin)

                    print(distances)
                    print(i_epoch, loss)
                    train_loss_list.append(loss.item())
                    if loss.item() == 0:
                        continue                
                    
                    if torch.isnan(loss):
                        print(distances)
                        continue
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_list.append(loss.item())
                
            print(f"train_loss : {sum(train_loss_list)/len(train_loss_list)}")
        print(valid_loss_list)

        # save loss
        if save_path is not None:
            loss_file = open(save_path + "/loss.pickle", 'wb')
            valid_loss_file = open(save_path + "/valid_loss.pickle", "wb")
            pickle.dump(loss_list, loss_file)
            pickle.dump(valid_loss_list, valid_loss_file)
            loss_file.close

        
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = SupervisedTree(10, 4, device=device)
    model.to(device)
    
    mass = torch.tensor([[0.4, 0.4, 0.1, 0.1],
                         [0.8, 0.2, 0.0, 0.0],
                         [0.5, 0.3, 0.2, 0.0],
                         [0.2, 0.5, 0.15, 0.15],
                         [0.1, 0.1, 0.4, 0.4],
                         [0.2, 0.0, 0.3, 0.5],
                         [0.1, 0.1, 0.8, 0.0]])
    labels = torch.tensor([1,1,1,1,2,2,2]).float()

    dataset = MyDataset(mass, labels)
    data_loader = DataLoader(dataset, batch_size=7, shuffle=True)
    mass = mass.to(device)
    print(model.calc_distances(mass))
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    model.run_train(data_loader, 200, optimizer, valid_loader=data_loader)
    mass = mass.to(device)
    print(model.calc_distances(mass))
    print(model.calc_ppar())
    
