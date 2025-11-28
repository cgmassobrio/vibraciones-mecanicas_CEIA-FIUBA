# Main frameworks
import numpy as np
import torch
import pandas as pd

# Data preparaTion
from torch.utils.data import Dataset
from scipy.stats import qmc


# ConfiguraTion of Datasets for Passing Through a Neural Network
class CustomDataset(Dataset):
    def __init__(self, I, O):
        super().__init__()
        self.I = I
        self.O = O

    def __len__(self):
        return self.I.shape[0]
    
    def __getitem__(self, idx):
        return self.I[idx, :], self.O[idx, :]


class DataSintetic:
    '''
    Class for creaTing datasets with placement points in the domain, boundary condiTions, and initial condiTions. 
    It also allows creaTing a dicTionary for training the network. Includes a method for plotTing placement points.
    '''
    def __init__(self, Xi, Xf, Ti, Tf, samples):
        '''
        Variables:
        - Xi, Xf: lower and upper bounds of the domain of posiTions.
        - Ti, Tf: lower and upper bounds of the domain of Time.
        - samples: total number of points for random sampling.
        '''

        self.Xi = Xi
        self.Xf = Xf
        self.Ti = Ti
        self.Tf = Tf
        self.samples = samples
        self.device = self._device()
        
        # Equally spaced tensor for space and Time domain
        self.x_tensor = torch.linspace(Xi, Xf, samples, device=self.device).view(-1,1)
        self.t_tensor = torch.linspace(Ti, Tf, samples, device=self.device).view(-1,1)

    # Device Assignment
    @staticmethod
    def _device():       
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        # Print devices, hardware, and CUDA version (if applicable)
        if device.type == 'cuda':
            print("".join(f'{d[0]}: {d[1]}\n' for d in [
                ('Device',device),
                ('Cluster',torch.cuda.get_device_name()),
                ('CUDA',torch.version.cuda),
            ]))
        else:
            print(f'DisposiTivo: {device}')

        return device    

    def puntual_data(self, N_p, x_i, sigma_i, LHS_seed_p):
        xi = x_i - sigma_i/2
        xf = x_i + sigma_i/2
        lb = np.array([xi, self.Ti])
        ub = np.array([xf, self.Tf])
        sampler = qmc.LatinHypercube(d=2, seed=LHS_seed_p)
        sample = sampler.random(n=N_p)
        return torch.from_numpy(qmc.scale(sample, lb, ub)).to(torch.float32).to(self.device)
        
    # Method for generaTing random placement points in the domain using LaTin Hypercube Sampling (LHS) for Timoshenko method
    def pde_T(self, N_pde, LHS_seed):
        '''
        Variables:
        - N_pde: total number of placement points for the domain of differenTial equaTions.
        - LHS_seed: seed for generaTing random values using LaTin Hypercube Sampling..
        '''    
        lb = np.array([self.Xi, self.Ti])
        ub = np.array([self.Xf, self.Tf])
        
        sampler = qmc.LatinHypercube(d=2, seed=LHS_seed)
        sample = sampler.random(n=N_pde)
        A_pde = torch.from_numpy(qmc.scale(sample, lb, ub)).to(torch.float32).to(self.device)
        eq_1 = torch.zeros((N_pde, 1), device=self.device)
        eq_2 = torch.zeros((N_pde, 1), device=self.device)
        D_pde = torch.cat([eq_1, eq_2], dim=1)
        return CustomDataset(A_pde,D_pde)
    
    # Method for generaTing random placement points in the domain using LaTin Hypercube Sampling (LHS) for estimate inverse function by Timoshenko method
    def pde_T_P(self, N_pde, LHS_seed, p_param, N_p, x_i, sigma_i, LHS_seed_p):
        '''
        Variables:
        - N_pde: total number of placement points for the domain of differenTial equaTions.
        - LHS_seed: seed for generaTing random values using LaTin Hypercube Sampling..
        '''
        lb = np.array([self.Xi, self.Ti])
        ub = np.array([self.Xf, self.Tf])
        
        sampler = qmc.LatinHypercube(d=2, seed=LHS_seed)
        sample = sampler.random(n=N_pde)
        A_pde = torch.from_numpy(qmc.scale(sample, lb, ub)).to(torch.float32).to(self.device)

        if np.any(p_param):
            idx_nz = np.flatnonzero(p_param)
            for i in idx_nz:
                A_p = self.puntual_data(N_p[i], x_i[i], sigma_i[i], LHS_seed_p[i])
                # A_p.unsqueeze(1) convierte A_p en forma [n, 1, 2], y A_pde.unsqueeze(0) en [1, m, 2]
                # Compara en una sola operación todas las filas sin importar la cantidad que tenga cada tensor
                compare = (A_p.unsqueeze(1) == A_pde.unsqueeze(0)) 
                coincidence = torch.all(compare, dim=2)  # verifica si las filas coinciden completamente
                mask = ~torch.any(coincidence, dim=1) # detecta si cada fila de A_p apareció en algún lado de A_pde, a la inversa (FALSE)
                A_p_filter = A_p[mask] # Se conservn las filas que NO aparecen en A_pde
                A_pde = torch.cat([A_pde, A_p_filter], dim=0)                

        eq_1 = torch.zeros((A_pde.size(0), 1), device=self.device)
        eq_2 = torch.zeros((A_pde.size(0), 1), device=self.device)
        D_pde = torch.cat([eq_1, eq_2], dim=1)
        return CustomDataset(A_pde,D_pde)
    

    # Generation of random placement points at the lower boundary of the domain for Timoshenko method
    def bc_l_T(self, N_bc_l, bc_l_seed):
        '''
        Variables:
        - N_bc_l: half of the total number of placement points for the lower boundary condiTions.
        - bc_l_seed: seed for generaTing random values for the lower boundary condiTion.
        '''
        rng_bc_l = np.random.default_rng(bc_l_seed)
        idx_bc_l = rng_bc_l.choice(self.t_tensor.size()[0], N_bc_l, replace=False)
        T_bc_l = self.t_tensor[idx_bc_l].view(-1,1).to(self.device)
        X_bc_l = self.Xi*torch.ones((N_bc_l, 1), device=self.device)
        A_bc_l = torch.cat([X_bc_l, T_bc_l], dim=1)
        eq_1 = torch.zeros((N_bc_l, 1), device=self.device)
        eq_2 = torch.zeros((N_bc_l, 1), device=self.device)
        D_bc_l = torch.cat([eq_1, eq_2], dim=1)
        return CustomDataset(A_bc_l, D_bc_l)
    
    # Generation of random placement points at the upper boundary of the domain for Timoshenko method
    def bc_u_T(self, N_bc_u, bc_u_seed):
        '''
        Variables:
        - N_bc_l: half of the total number of placement points for the upper boundary condiTions.
        - bc_l_seed: seed for generaTing random values for the upper boundary condiTion.
        '''
        rng_bc_u = np.random.default_rng(bc_u_seed)
        idx_bc_u = rng_bc_u.choice(self.t_tensor.size()[0], N_bc_u, replace=False)
        T_bc_u = self.t_tensor[idx_bc_u].view(-1,1).to(self.device)
        X_bc_u = self.Xf*torch.ones((N_bc_u, 1), device=self.device)
        A_bc_u = torch.cat([X_bc_u, T_bc_u], dim=1)
        eq_1 = torch.zeros((N_bc_u, 1), device=self.device)
        eq_2 = torch.zeros((N_bc_u, 1), device=self.device)
        D_bc_u = torch.cat([eq_1, eq_2], dim=1)
        return CustomDataset(A_bc_u, D_bc_u)
        
    # Generation of random placement points at the initial condiTions for Timoshenko method
    def ic_T(self, N_ic, ic_seed):
        '''
        Variables:
        - N_ic: total number of placement points for the initial conditions.
        - ic_seed: seed for generaTing random values for the initial conditions.
        '''
        rng_ic = np.random.default_rng(ic_seed)
        idx_ic = rng_ic.choice(self.x_tensor.size()[0], N_ic, replace=False)
        T_ic = self.Ti*torch.ones((N_ic, 1), device=self.device)
        X_ic = self.x_tensor[idx_ic].view(-1,1).to(self.device)
        A_ic = torch.cat([X_ic, T_ic], dim=1)
        eq_1 = torch.zeros((N_ic, 1), device=self.device)
        eq_2 = torch.zeros((N_ic, 1), device=self.device)
        eq_3 = torch.zeros((N_ic, 1), device=self.device)
        eq_4 = torch.zeros((N_ic, 1), device=self.device)
        D_ic = torch.cat([eq_1, eq_2, eq_3, eq_4], dim=1)
        return CustomDataset(A_ic, D_ic)
    
    # Generation of random placement points by sensor loc for calc inverse problem Timoshenko beam
    # def inv_data(self, N_data, omega, sensor_loc, inv_seed):
        '''
        Variables:
        N_data:
        omega: 
        sensor_loc:
        inv_seed:
        '''
        # X_data, T_data = [], []
        # for loc, seed in zip(sensor_loc, inv_seed):
        #     rng_data = np.random.default_rng(seed)
        #     idx_data = rng_data.choice(self.x_tensor.size()[0], N_data, replace=False)
        #     X_data.append(torch.full((N_data, 1), loc, device=self.device))
        #     T_data.append(self.t_tensor[idx_data].view(-1,1).to(self.device))
        
        # A_data = torch.cat([torch.cat(X_data, dim=0), torch.cat(T_data, dim=0)], dim=1)
        # W = ((np.pi/2)*torch.sin(A_data[:, 0:1])*torch.cos(A_data[:, 1:2])).to(self.device)
        # W_data = W + omega*torch.max(W).to(self.device)*torch.normal(mean=0.0, std=1.0, size=(len(sensor_loc)*N_data,1)).to(self.device)
        # Psi = (torch.cos(A_data[:, 1:2])*((torch.pi/2)*torch.cos(A_data[:, 0:1]) + (A_data[:, 0:1] - np.pi/2))).to(self.device)
        # Psi_data = Psi + omega*torch.max(Psi).to(self.device)*torch.normal(mean=0.0, std=1.0, size=(len(sensor_loc)*N_data, 1)).to(self.device)
        # D_data = torch.cat([W_data, Psi_data], dim=1)
        # return CustomDataset(A_data, D_data)

    def inv_data(self, N_data, omega, sensor_loc, inv_seed, df):
        '''
        Variables:
        N_data: cantidad de datos por sensor
        omega: nivel de ruido
        sensor_loc: lista con 9 valores de x
        inv_seed: lista de semillas para reproducibilidad
        DataFrame con columnas ['x', 't', 'displacements', 'rotations']
        '''
        
        X_data, T_data, W_data, Psi_data = [], [], [], []
        for loc, seed in zip(sensor_loc, inv_seed):
            rng_data = np.random.default_rng(seed)

            # Buscar el valor de x más cercano a loc
            X_vals_df = df['x'].unique()
            X_closest = X_vals_df[np.argmin(np.abs(X_vals_df - loc))]

            # Filtrar el DataFrame por ese valor de x
            df_sensor = df[df['x'] == X_closest]

            # Seleccionar N_data muestras aleatorias de ese sensor
            seed_int = rng_data.integers(0, 1e6)
            df_sample = df_sensor.sample(n=N_data, random_state=seed_int)

            # Extraer columnas como tensores
            X_vals = torch.tensor(df_sample['x'].values/1, dtype=torch.float32).view(-1, 1).to(self.device)
            T_vals = torch.tensor(df_sample['t'].values/0.17379, dtype=torch.float32).view(-1, 1).to(self.device)
            W_vals = torch.tensor(df_sample['displacements'].values/1, dtype=torch.float32).view(-1, 1).to(self.device)
            Psi_vals = torch.tensor(df_sample['rotations'].values, dtype=torch.float32).view(-1, 1).to(self.device)

            # Agregar ruido
            w_noise = omega * torch.max(W_vals).to(self.device) * torch.normal(mean=0.0, std=1.0, size=W_vals.shape).to(self.device)
            psi_noise = omega * torch.max(Psi_vals).to(self.device) * torch.normal(mean=0.0, std=1.0, size=Psi_vals.shape).to(self.device)

            W_data.append(W_vals + w_noise)
            Psi_data.append(Psi_vals + psi_noise)
            X_data.append(X_vals)
            T_data.append(T_vals)

        # Concatenar todos los datos
        A_data = torch.cat([torch.cat(X_data, dim=0), torch.cat(T_data, dim=0)], dim=1)
        W_data = torch.cat(W_data, dim=0)
        Psi_data = torch.cat(Psi_data, dim=0)
        D_data = torch.cat([W_data, Psi_data], dim=1)

        return CustomDataset(A_data, D_data)



    
    # Compilation of PDE datasets, boundary conditions, and initial conditions for Timoshenko method
    def ds_train_T(self, N_pde, N_bc_l, N_bc_u, N_ic, LHS_seed, bc_l_seed, bc_u_seed, ic_seed, 
                   N_data=None, omega=None, sensor_loc=None, inv_seed=None,  
                   p_param=None, N_p=None, x_i=None, sigma_i=None,  LHS_seed_p=None,
                   isData=False, df=None, isP=True, save_data=False, save_data_path=None):
        '''
        AdiTionals variables:
        - stage: desTinaTion of the datasets for 'train', 'validate', or 'test' (currently only 'train' is enabled)
        - save_data: indicator to specify whether to save the dicTionary or not. False by default.
        - isData: indicator to specify whether the analysis will use labeled data or not. False by default.
        '''
        # References to each dataset
        tags = ['PDE', 'BoundaryConditionsLower', 'BoundaryConditionsUpper', 'InitialCondition']

        # Initialization of datasets list
        if isP:
            ds = [self.pde_T_P(N_pde, LHS_seed, p_param, N_p, x_i, sigma_i, LHS_seed_p), 
                  self.bc_l_T(N_bc_l, bc_l_seed), self.bc_u_T(N_bc_u, bc_u_seed), self.ic_T(N_ic, ic_seed)]
        else:
            ds = [self.pde_T(N_pde, LHS_seed), self.bc_l_T(N_bc_l, bc_l_seed), self.bc_u_T(N_bc_u, bc_u_seed), self.ic_T(N_ic, ic_seed)]
            
        if isData:
            tags.append("LabelledData")
            ds.append(self.inv_data(N_data, omega, sensor_loc, inv_seed, df))

        # Dictionary with the datasets
        data_dict = dict(zip(tags,ds))

        # Save dictionary
        if save_data:
            torch.save(data_dict, save_data_path)
            print('Dataset saved successfully')

        return data_dict
    
    # Generation of points to predict W and Psi
    def ds_predict(self):
        return torch.cat([self.x_tensor, self.t_tensor], dim=1)
    