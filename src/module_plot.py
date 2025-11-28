import numpy as np
import torch
from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Line3DCollection

class Plot():
    '''
    Class for plotting training datasets, losses, and predictions
    '''
    # 2D mesh plot for predict.
    def plot_predict2D(self, data_dict, figure_path, level_cont=None, title=None):
        # Domain in cpu opetation
        X = data_dict['input'][:, 0:1]
        T = data_dict['input'][:, 1:2]
        O = data_dict['output']
        X_m, T_m = np.meshgrid(X, T)
        # Graphics
        fig, ax = plt.subplots(figsize=(10,5))
        heatmap = ax.contourf(T_m, X_m, O, levels=level_cont, cmap='rainbow')
        cbar = fig.colorbar(heatmap, ax=ax, orientation='vertical')
        ax.set_xlabel('T')
        ax.set_ylabel('X')
        ax.set_title(title)
        plt.savefig(figure_path)
        plt.close(fig)
        print('Graph saved correctly')

    # 3D mesh plot for predict.
    def plot_predict3D(self, data_dict, figure_path, level_cont=None, title=None):
        # Domain in cpu opetation
        X = data_dict['input'][:, 0:1]
        T = data_dict['input'][:, 1:2]
        O = data_dict['output']
        X_m, T_m = np.meshgrid(X, T)
        # Graphics
        fig, ax = plt.subplots(figsize=(10,5), subplot_kw={"projection": "3d"})
        heatmap = ax.plot_surface(T_m, X_m, O, levels=level_cont, cmap='rainbow')
        cbar = fig.colorbar(heatmap, ax=ax, orientation='vertical')
        ax.set_xlabel('T')
        ax.set_ylabel('X')
        ax.set_title(title)
        plt.savefig(figure_path)
        plt.close(fig)
        print('Graph saved correctly')

    # Collocation points plot.
    def plot_train(self, data_dict, figure_path, isData=False, title=None):
        # Operation in CPU domain
        A_pde = data_dict['PDE'].I   
        A_pde_cpu = A_pde.cpu().numpy()   
        # Operation in CPU boundary conditions
        A_bc_l = data_dict['BoundaryConditionsLower'].I
        A_bc_u = data_dict['BoundaryConditionsUpper'].I
        A_bc = torch.cat([A_bc_l, A_bc_u], dim=0)
        A_bc_cpu = A_bc.cpu().numpy()
        # Operation in CPU initial conditions
        A_ic = data_dict['InitialCondition'].I
        A_ic_cpu = A_ic.cpu().numpy()
        # Plots
        fig, ax = plt.subplots(1,1, figsize=(10, 6))
        ax.scatter(A_pde_cpu[:, 1:2], A_pde_cpu[:, 0:1], s=4, c='r', label=r'$N_{dom}$='f'{len(A_pde_cpu)}')
        ax.scatter(A_bc_cpu[:, 1:2], A_bc_cpu[:, 0:1], s=4, c='b', label=r'$N_{bc}$='f'{len(A_bc_cpu)}')
        ax.scatter(A_ic_cpu[:, 1:2], A_ic_cpu[:, 0:1], s=4, c='g', label=r'$N_{ic}$='f'{len(A_ic_cpu)}')
        if isData:
            A_data = data_dict['LabelledData'].I
            A_data_cpu = A_data.cpu().numpy()
            ax.scatter(A_data_cpu[:, 1:2], A_data_cpu[:, 0:1], s=4, c='m', label=r'$N_{data}$='f'{len(A_data_cpu)}')

        ax.grid()
        ax.set_title(title)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        plt.xlabel('Dominio del tiempo (T)')
        plt.ylabel('Dominio del espacio (X)')            
        plt.legend(bbox_to_anchor=(1.04,1), loc='upper left')
        plt.savefig(figure_path)
        plt.close(fig)
        print('Graph saved correctly')

    # Loss plot
    def loss_graphics(self, loss_train_dict, figure_path):
        # The variable loss_train_dict is a dictionary with the lists of total losses (overall), PDE, BC (boundary conditions), and IC (initial conditions)
        loss_train_list = list(loss_train_dict.values())
        titles = list(loss_train_dict.keys())

        colors = ["blue", "orange", "green", "red", "purple", "brown"]

        fig, ax = plt.subplots(1, len(loss_train_list), figsize=(28,5))
        for i in range(len(loss_train_list)):
            epochs_train = np.arange(0, len(loss_train_list[i]))
            ax[i].loglog(epochs_train, loss_train_list[i], colors[i])
            ax[i].grid()
            ax[i].set_xlabel('Epochs')
            ax[i].set_title(titles[i])
        plt.savefig(figure_path)
        plt.close(fig)

    # Coeficients to inverse problem
    def coeficients_graphics(self, coeficients_values_dict, figure_path):

        coeficients_values_list = list(coeficients_values_dict.values())
        titles = list(coeficients_values_dict.keys())
        
        colors = ["blue", "orange", "green", "red", "purple", "brown"]


        fig, ax = plt.subplots(1, len(coeficients_values_list), figsize=(25,5))
        if len(coeficients_values_list) == 1:
            epochs_train = np.arange(0, len(coeficients_values_list[0]))
            ax.semilogy(epochs_train, coeficients_values_list[0], colors[0])
            ax.grid()
            ax.set_xlabel('Epochs')
            ax.set_title(titles[0])
        else:
            for i in range(len(coeficients_values_list)):
                epochs_train = np.arange(0, len(coeficients_values_list[i]))
                ax[i].semilogy(epochs_train, coeficients_values_list[i], colors[i])
                ax[i].grid()
                ax[i].set_xlabel('Epochs')
                ax[i].set_title(titles[i])
        plt.savefig(figure_path)
        plt.close(fig)



