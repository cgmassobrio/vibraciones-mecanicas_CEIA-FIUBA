# Main frameworks
import numpy as np
import torch
from torch import nn 
from torch import optim

# Functionalities
import json
import os

# Modulus
import sys 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.module_data import DataSintetic
from src.module_pinn import Fit_model
from src.module_plot import Plot

# Path directories
PARAMS_PATH = 'parameters'
DATA_PATH = 'data'
MODELS_PATH = 'models'
METRICS_PATH = 'metrics'
FIGURES_PATH = 'figures'

# Import files names
param_data_file = 'param_data-predict.json'
param_model_file = 'param_model.json'
param_train_file = 'param_train.json'
data_file = 'train_test.pt'

# Save files names
model_file = 'model_test_29.pt'
loss_file = 'loss_test_29.pt'
fig_loss_file = 'loss_test_29.png'


# Model definition
param_data_path = os.path.join(parent_dir, PARAMS_PATH, param_data_file)
param_model_path = os.path.join(parent_dir, PARAMS_PATH, param_model_file)
vib_model = Fit_model(param_data_path, param_model_path)

#--------------- Train model-----------------#
# Reading JSON with training parameters
param_train_path = os.path.join(parent_dir, PARAMS_PATH, param_train_file)
with open(param_train_path, 'r') as f:
    param_train = json.load(f)

# Load datasets
data_path = os.path.join(parent_dir, DATA_PATH, data_file)
datasets = torch.load(data_path)

# Directories to save model and metric files
model_path = os.path.join(parent_dir, MODELS_PATH, model_file)
loss_path = os.path.join(parent_dir, METRICS_PATH, loss_file)

# Weight initialization and definition of loss function and optimizer
vib_model.init_xavier_un()
loss_fn = nn.MSELoss()

# Define optimizer
optimizer = optim.LBFGS(vib_model.model.parameters(), lr=param_train['lr'], max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)

# Train model
vib_model.structureView()
vib_model.train(**param_train, datasets=datasets, loss_fn=loss_fn, optimizer=optimizer, save_model=True, model_path=model_path, metric_path=loss_path)

# Loss plot
fig_loss_path = os.path.join(parent_dir, FIGURES_PATH, fig_loss_file)
loss_plot = Plot()
loss_plot.loss_graphics(vib_model.loss_train_dict, fig_loss_path)

