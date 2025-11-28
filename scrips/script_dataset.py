
# Funcionalities
import json
import os
from datetime import datetime

# Modulus 
import sys 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.module_data import DataSintetic
from src.module_plot import Plot

# Path directories
PARAMS_PATH = 'parameters'
DATA_PATH = 'data'
FIGURES_PATH = 'figures'

# Import files names
param_data_file = 'param_data-predict.json'
param_model_file = 'param_model.json'

# Save files names
data_file = 'train_T-inv-func.pt'
fig_train_file = 'train_T-inv-func.png'


##--------------------------------------------------------------------##
##--------------------------------------------------------------------##
# runtime count inicialization
t0 = datetime.now()

# Reading JSON with data parameters
param_data_path = os.path.join(parent_dir, PARAMS_PATH, param_data_file)
with open(param_data_path, 'r') as f:
    param_data = json.load(f)
PDE = param_data['PDE']
conditions = param_data['conditions']

# Check JSON with inverse data parameters
isData = True
isP = False
if isData:
      inverse = param_data['inverse']

if isP:
     conditions = param_data['p_conditions']

# Dataset generation
data_file_path = os.path.join(parent_dir, DATA_PATH, data_file)
points = DataSintetic(**PDE)

# Colocation point plot 
fig_train_path = os.path.join(parent_dir, FIGURES_PATH, fig_train_file)
data_plot = Plot()

# Select process type
if isData:
    data_train = points.ds_train_T(**conditions, **inverse, isData=True, save_data=True, save_data_path=data_file_path)
    data_train_plot = data_plot.plot_train(data_train, fig_train_path, isData=True)
else:
    data_train = points.ds_train_T(**conditions, save_data=True, save_data_path=data_file_path)
    data_train_plot = data_plot.plot_train(data_train, fig_train_path)       

elapsed_time = datetime.now() - t0
print('\n Dataset time:', elapsed_time.seconds,'s')  