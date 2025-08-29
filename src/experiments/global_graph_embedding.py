from torch.utils.data import DataLoader
import torch
import random
import numpy as np
from src.data.data_loader import dataframe_to_dataset
import pandas as pd
from src.utils.train import train
from sklearn.model_selection import train_test_split
import os


def run_experiment(data_folder_path, config_folder_path, hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_path = os.path.join(data_folder_path, "rotterdam_data.csv")
    # data_df = pd.read_csv(data_path)
    # train_df,test_df = train_test_split(data_df, test_size=0.3, random_state=42)
    train_path = os.path.join(data_folder_path, "train.csv")
    test_path = os.path.join(data_folder_path, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    

    edge_path = os.path.join(data_folder_path, "buurt_adjacency.csv")
    edge_ind = pd.read_csv(edge_path,index_col=0)
    
    config_path = os.path.join(config_folder_path,"data","neighbor_graph.json")
    train_dataset, test_dataset = dataframe_to_dataset(config_path, train_df, test_df)

    
    
    train(config_path, edge_ind, train_dataset, test_dataset, device, hyperparameters)

def run_dynamic_baseline_experiment(data_folder_path, config_folder_path, hyperparameters):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    neighborhood_feature_path = os.path.join(data_folder_path, "neighborhoods.csv")
    train_path = os.path.join(data_folder_path, "train.csv")
    test_path = os.path.join(data_folder_path, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    