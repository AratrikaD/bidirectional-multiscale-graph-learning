import torch
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((np.exp(y_true)- np.exp(y_pred)) / (np.exp(y_true) + 1e-8))) * 100