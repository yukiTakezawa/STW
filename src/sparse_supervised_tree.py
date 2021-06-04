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
from supervised_tree import *


class SparseSupervisedTree(torch.nn.Module):
    """
    SupervisedTree for Evaluation. 
    """
    
    def __init__(self, supervised_tree, device='cpu'):
        super(SparseSupervisedTree, self).__init__()
        self.n_inner = supervised_tree.n_inner # number of inner nodes which contained root.
        self.n_leaf = supervised_tree.n_leaf # number of leaf nodes. (number of words)
        self.n_node = self.n_leaf + self.n_inner # number of all nodes.
        self.device = device
               
        self.param = supervised_tree.param.data.to(device)

        self.A = self.gen_A()        
        self.inv_A = self.calc_inv_A()

        self.block_psub = self.calc_psub()
    
    def gen_A(self):
        A = torch.zeros(self.n_inner, self.n_inner-1)
        for i in range(1, int(self.n_inner/5)):
            A[i-1, 5*(i-1):5*i] = 1.0
        A[int(self.n_inner/5)-1, 5*(int(self.n_inner/5)-1):] = 1.0
        A = A.to(self.device)
        return torch.eye(self.n_inner, device=self.device) - torch.cat([torch.zeros(self.n_inner, 1, device=self.device), A], dim=1).to(self.device)

    
    def calc_inv_A(self):
        return self.A.inverse()

    
    def calc_ppar(self):
        exp_param = F.softmax(self.param, dim=0)
        b = exp_param.max(0, keepdim=True)[0]
        exp_param = exp_param.ge(b).float()
        return torch.cat([torch.eye(self.n_inner, device=self.device) - self.A, exp_param], dim=1)

    
    def calc_psub(self, block_D=None):
        """
        Retuens
        ----------
        X : torch.tensor (shape is (self.n_inner, self.n_leaf))
            If j-th word is contained in the subtree rooted at v_i, then X[i][j] = 1, otherwise 0.
        """

        B = F.softmax(self.param, dim=0)

        b = B.max(0, keepdim=True)[0]
        B = B.ge(b).float()
        
        X = torch.mm(self.inv_A, B)
        return X

    
    def calc_distance(self, mass1, mass2):
        return torch.abs(torch.mv(self.block_psub, mass1 - mass2)).sum() + torch.abs(mass1 - mass2).sum()

    
    def calc_distance2(self, mass1, batch_mass):
        return torch.abs(torch.mm(self.block_psub, (batch_mass - mass1).T)).sum(dim=0) + torch.abs(batch_mass - mass1).sum(1)

    
    def calc_distances(self, mass):
        """
        Parameters
        ----------
        mass : torch.tensor (shape is (n_doc, self.n_leaf))
        
        Returns
        ----------
        distances : torch.tensor (shape is (n_doc, n_doc))
            distance[i][j] is tree-wasserstein distance between mass[i] and mass[j].
        """

        distances = torch.zeros((mass.shape[0], mass.shape[0]), device=self.device)
        for i in range(mass.shape[0]):
            for j in range(i):
                distances[i][j] = self.calc_distance(mass[i], mass[j])

        return distances + distances.permute(1,0)

    
    def forward(self, mass):
        """
        Parameters
        ----------
        mass : torch.tensor (shape is (n_doc, self.n_leaf))
            mass[i] is mass of i-th document.

        
        Returns
        ----------
        distances : torch.tensor (shape is (n_doc, n_doc))
            distance[i][j] is tree-wasserstein distance between i-th and j-th document.
        """
        
        return self.calc_distances(mass)


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
    """
    mass = torch.tensor([[0.4, 0.0, 0.4, 0.2],
                         [0.8, 0.0, 0.0, 0.2],
                         [0.1, 0.0, 0.8, 0.1],
                         [0.5, 0.0, 0.1, 0.4],
                         [0.5, 0.5, 0.0, 0.0],
                         [0.0, 0.6, 0.4, 0.0],
                         [0.0, 0.7, 0.2, 0.1]])
    """
    
    labels = torch.tensor([1,1,1,1,2,2,2]).float()

    dataset = MyDataset(mass, labels)
    data_loader = DataLoader(dataset, batch_size=7, shuffle=True)
    #print(model.calc_distances(mass))
    mass = mass.to(device)
    print(model.calc_distances(mass))
    
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    model.run_train(data_loader, 200, optimizer, valid_loader=data_loader)
    mass = mass.to(device)
    print(model.calc_distances(mass))
    print(model.calc_ppar())

    sparse_model = SparseSupervisedTree(model, device=device)
    print("calc_ppar")
    print(model.calc_ppar())
    print(sparse_model.calc_ppar())

    print("calc_psub")
    print(model.calc_psub())
    print(sparse_model.calc_psub())

    print(model.calc_distances(mass))
    print(sparse_model.calc_distances(mass))
