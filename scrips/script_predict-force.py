import torch

# Funcionalities
import json
import os

# Modulus
import sys 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from src.module_data import *
from src.module_pinn_func import Fit_inverse
# from src.module_pinn_inv import Fit_inverse
from src.module_plot import Plot


# Path directories
PARAMS_PATH = 'parameters'
DATA_PATH = 'data'
MODELS_PATH = 'models'
FIGURES_PATH = 'figures'

# Import files names
param_data_file = 'param_data-predict.json'
param_model_file = 'param_model.json'
# model_file = 'model_force.pt'

# Save files names
fig_predict_file_F1 = 'predict_F1.png'
fig_error_file_F1 = 'error_F1.png'

# Path files names
param_data_path = os.path.join(parent_dir, PARAMS_PATH, param_data_file)
fig_predict_path_F1 = os.path.join(parent_dir, FIGURES_PATH, fig_predict_file_F1)
fig_error_path_F1 = os.path.join(parent_dir, FIGURES_PATH, fig_error_file_F1)

##--------------------------------------------------------------------##
##--------------------------------------------------------------------##
# Reading JSON with data parameters
with open(param_data_path, 'r') as f:
    data = json.load(f)
PDE = data['PDE']

# Dataset building for test
ds_pred = DataSintetic(**PDE)
A = ds_pred.ds_predict()
X_m, T_m = torch.meshgrid(A[:, 0], A[:, 1], indexing='xy')

# Coeficients of inverse problem
beta = -0.04791
gamma = 0.77239
theta = 1.30509

# Domain in cpu and numpy operation 
A_cpu = A.cpu().numpy()
X_m_cpu = X_m.cpu().numpy()
T_m_cpu = T_m.cpu().numpy()

# Estimate solution
F1_est = -gamma*np.ones_like(X_m_cpu) + theta*np.sin(X_m_cpu)*np.cos(T_m_cpu)

# Exact solution
F1 = -np.cos(T_m_cpu) + (np.pi/2)*np.sin(X_m_cpu)*np.cos(T_m_cpu)

# Estimation error
est_error_F1_ecm = np.mean((F1_est.flatten() - F1.flatten())**2)/np.mean(F1.flatten()**2)
est_error_F1_n2 = np.linalg.norm((F1_est.flatten() - F1.flatten()), ord=2)/np.linalg.norm(F1.flatten(), ord=2)
print(f'Error estimado de predicción de la fuerza de excitación F1 (ECM): {est_error_F1_ecm}')
print(f'Error estimado de predicción de la fuerza de excitación F1 (norma 2): {est_error_F1_n2}')

predict_plot = Plot()

# Prediction plot
data_pred_dict_F2 = {'input': A_cpu, 'output': F1_est}
predict_plot.plot_predict2D(data_pred_dict_F2, fig_predict_path_F1, level_cont=25, title='Predicción de la fuerza de excitación F1')

# Diference error plot
abs_er_F1 = F1-F1_est
data_abs_error_dict_F1 = {'input': A_cpu, 'output': abs_er_F1}
predict_plot.plot_predict2D(data_abs_error_dict_F1, fig_error_path_F1, level_cont=30, title='Diferencia de la fuerza de excitación F1')

