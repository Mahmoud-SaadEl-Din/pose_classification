import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from os.path import join
import csv
import pandas as pd

class NN_pose_dataset(Dataset):
    def __init__(self, data_dir):
        self.data = pd.read_csv(data_dir,delimiter=',')
        self.size = len(self.data.index) 

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        row = self.data.loc[idx].to_list()   
        return torch.FloatTensor(row[2:26]), 0 if row[1]=="pose" else 1  # Assuming filenames are unique identifiers


# Specify the path to your test data
NN_dataloaders = {}
NN_dataset_sizes = {}
for set in ["train","val"]:
    data_dir = f'small_{set}_poses_normalized_no_feature_selection.csv'

    # Create a custom test dataset
    dataset = NN_pose_dataset(data_dir)
    NN_dataset_sizes[set] = dataset.size
    # Create a data loader for the test dataset
    NN_dataloaders[set] = DataLoader(dataset, batch_size=1, shuffle=True)