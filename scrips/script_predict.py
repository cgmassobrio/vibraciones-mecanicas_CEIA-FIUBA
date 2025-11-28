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
model_file = 'trained_model_T-inv-func_SS-SS.pt'

# Save files names
fig_predict_file_W = 'predict_T-inv-func_W_SS-SS.png'
fig_error_file_W = 'error_T-inv-func_W_SS-SS.png'
fig_predict_file_Psi = 'predict_T-inv-func_Psi_SS-SS.png'
fig_error_file_Psi = 'error_T-inv-func_Psi_SS-SS.png'


# Path files names
param_data_path = os.path.join(parent_dir, PARAMS_PATH, param_data_file)
param_model_path = os.path.join(parent_dir, PARAMS_PATH, param_model_file)
model_path = os.path.join(parent_dir, MODELS_PATH, model_file)
fig_predict_path_W = os.path.join(parent_dir, FIGURES_PATH, fig_predict_file_W)
fig_error_path_W = os.path.join(parent_dir, FIGURES_PATH, fig_error_file_W)
fig_predict_path_Psi = os.path.join(parent_dir, FIGURES_PATH, fig_predict_file_Psi)
fig_error_path_Psi = os.path.join(parent_dir, FIGURES_PATH, fig_error_file_Psi)

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

model = Fit_inverse(param_model_path)
A_flatten = torch.cat([X_m.flatten().reshape(-1,1), T_m.flatten().reshape(-1,1)], dim=1)
D = model.predict(A_flatten, model_path)

# Domain in cpu and numpy operation 
A_cpu = A.cpu().numpy()
X_m_cpu = X_m.cpu().numpy()
T_m_cpu = T_m.cpu().numpy()
D_cpu = D.cpu().numpy()

W = (np.pi/2)*np.sin(X_m_cpu)*np.cos(T_m_cpu)
Psi = np.cos(T_m_cpu)*((np.pi/2)*np.cos(X_m_cpu) + (X_m_cpu - np.pi/2))

# Exact solution
W = (np.pi/2)*np.sin(X_m_cpu)*np.cos(T_m_cpu)
Psi = np.cos(T_m_cpu)*((np.pi/2)*np.cos(X_m_cpu) + (X_m_cpu - np.pi/2))

# Estimation error
est_error_W_ecm = np.mean((D_cpu[:, 0] - W.flatten())**2)/np.mean(W.flatten()**2)
est_error_W_n2 = np.linalg.norm((D_cpu[:, 0] - W.flatten()), ord=2)/np.linalg.norm(W.flatten(), ord=2)
print(f'Error estimado de predicción de desplazamientos W (ECM): {est_error_W_ecm}')
print(f'Error estimado de predicción de desplazamientos W (norma 2): {est_error_W_n2}')

est_error_Psi_ecm = np.mean((D_cpu[:, 1] - Psi.flatten())**2)/np.mean(Psi.flatten()**2)
est_error_Psi_n2 = np.linalg.norm((D_cpu[:, 1] - Psi.flatten()), ord=2)/np.linalg.norm(Psi.flatten(), ord=2)
print(f'Error estimado de predicción de desplazamientos Psi (ECM): {est_error_Psi_ecm}')
print(f'Error estimado de predicción de desplazamientos Psi (norma 2): {est_error_Psi_n2}')

W_pred_mesh = D_cpu[:, 0].reshape(-A_cpu.shape[0], A_cpu.shape[0])
Psi_pred_mesh = D_cpu[:, 1].reshape(-A_cpu.shape[0], A_cpu.shape[0])

predict_plot = Plot()

# Prediction plot
data_pred_dict_W = {'input': A_cpu, 'output': W_pred_mesh}
predict_plot.plot_predict2D(data_pred_dict_W, fig_predict_path_W, level_cont=25, title='Predicción de desplazamiento W')

# Prediction plot
data_pred_dict_Psi = {'input': A_cpu, 'output': Psi_pred_mesh}
predict_plot.plot_predict2D(data_pred_dict_Psi, fig_predict_path_Psi, level_cont=25, title='Predicción de desplazamiento Psi')

# Absolute error plot
abs_er_W = W-W_pred_mesh
data_abs_error_dict_W = {'input': A_cpu, 'output': abs_er_W}
predict_plot.plot_predict2D(data_abs_error_dict_W, fig_error_path_W, level_cont=30, title='Diferencia de desplazamiento W')

# Diference error plot
abs_er_Psi = Psi-Psi_pred_mesh
data_abs_error_dict_Psi = {'input': A_cpu, 'output': abs_er_Psi}
predict_plot.plot_predict2D(data_abs_error_dict_Psi, fig_error_path_Psi, level_cont=30, title='Diferencia de desplazamiento Psi')

