import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class GraphDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        dataset = fname.split('/')[-3]

        data = np.load(fname)
        if dataset == 'bandlimited' or dataset == 'swissroll':
            data_node_obs = data['data_node_obs'][:,np.newaxis]
            data_node_gt = data['data_node'][:,np.newaxis]
            L = data['L']
            A = data['A']
        if dataset == 'human':
            data_node_obs = data['data_color_obs']
            data_node_gt = data['data_color']
            L = data['L']
            A = data['A']
        if dataset == 'modelnet':
            data_node_obs = data['data_node_obs']
            data_node_gt = data['data_node']
            L = data['L']
            A = data['A']
        if dataset == 'sst':
            data_node_obs = data['data_node_obs'][:,np.newaxis]
            data_node_gt = data['data_node'][:,np.newaxis]
            L = data['L']
            A = data['A']
        
        # Convert to torch tensors and move to device if necessary
        # Note: device handling will be done in train_proposed.py
        return (torch.tensor(data_node_gt, dtype=torch.float64),
                torch.tensor(data_node_obs, dtype=torch.float64),
                torch.tensor(L, dtype=torch.float64),
                torch.tensor(A, dtype=torch.float64))

def load_data(fname):
    # This function will still be used by GraphDataset internally
    # but direct calls to it will be replaced by DataLoader usage.
    dataset = fname.split('/')[-3]

    data = np.load(fname)
    if dataset == 'bandlimited' or dataset == 'swissroll':
        data_node_obs = data['data_node_obs'][:,np.newaxis]
        data_node_gt = data['data_node'][:,np.newaxis]
        L = data['L']
        A = data['A']
    if dataset == 'human':
        data_node_obs = data['data_color_obs']
        data_node_gt = data['data_color']
        L = data['L']
        A = data['A']
    if dataset == 'modelnet':
        data_node_obs = data['data_node_obs']
        data_node_gt = data['data_node']
        L = data['L']
        A = data['A']
    if dataset == 'sst':
        data_node_obs = data['data_node_obs'][:,np.newaxis]
        data_node_gt = data['data_node'][:,np.newaxis]
        L = data['L']
        A = data['A']
    
    return data_node_gt, data_node_obs, L, A
