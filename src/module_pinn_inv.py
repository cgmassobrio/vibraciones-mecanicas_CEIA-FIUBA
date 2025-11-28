# Main frameworks
import torch

# Neural network modeling
from torch import nn

# Files/Functionalities
# import sys 
# sys.path.append('../')
import json
from datetime import datetime



class PINN_inverse(nn.Module):
    '''
    Class that defines the PINN model
    Input:
    - param_model: JSON file containing dictionaries that group the parameters needed to work with the PINN model.
    '''

    def __init__(self, param_data, param_model):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        '''
        Open JSON. Dictionaries:
        - physics: physics model parameters.
        - model_nn: architecture parameters and the forward stage
        '''
        with open(param_data, 'r') as f:
            data_domain = json.load(f)
        PDE = data_domain['PDE']

        with open(param_model, 'r') as f:
            data_model = json.load(f)
        physics = data_model['physics']
        model_nn = data_model['parameters']

        self.Tf = PDE['Tf']    

        # Physics model parameters
        self.R02 = physics['R02']
        self.gamma_bs = physics['gamma_bs']
        self.x_i = physics['x_i']
        self.K_L = physics['K_L']
        self.K_R = physics['K_R']
        self.K_tL = physics['K_tL']
        self.K_tR = physics['K_tR']
        self.Fa_m = physics['Fa_m']
        self.C = physics['C']
        self.M_1 = physics['M_1']
        self.K_1 = physics['K_1']
        self.K_t1 = physics['K_t1']
        self.P_av = physics['P_av']
        self.sigma_i = physics['sigma_i']
        self.Fa_p = physics['Fa_p']
        
        # Architecture parameters
        self.LB = model_nn['LB'] # 0.0
        self.LU = model_nn['LU'] # 1.0
        self.InputDimension = model_nn['InputDimension'] 
        self.OutputDimension = model_nn['OutputDimension'] 
        self.NumberOfNeurons = model_nn['NumberOfNeurons'] 
        self.NumberOfHiddenLayers = model_nn['NumberOfHiddenLayers']
        self.ActivationFunction = eval(model_nn['ActivationFunction']) # Tanh


        # unknown coefficient declared as a network parameter 
        # self.alpha_inv_input = nn.Parameter(torch.tensor(1.0))
        # self.UnknownCoeficient = self.alpha_inv_input

        # Architecture definition 
        self.InputLayer = nn.Linear(self.InputDimension, self.NumberOfNeurons)
        self.HiddenLayers = nn.ModuleList(
            [nn.Linear(self.NumberOfNeurons, self.NumberOfNeurons) for _ in range(self.NumberOfHiddenLayers - 1)])
        self.OutputLayer = nn.Linear(self.NumberOfNeurons, self.OutputDimension)

    def forward(self, input):
        input = 2*(input - self.LB)/(self.LU - self.LB) - 1 # Renormalización  [-1,1] 
        output = self.ActivationFunction(self.InputLayer(input))
        for k, l in enumerate(self.HiddenLayers):
            output = self.ActivationFunction(l(output))
        output = self.OutputLayer(output)
        return output

    # Calculation of the governing equations, as residuals of the PINN model
    def governingEquationResidue(self, A, B):
        # Inputs:
        # - A: Collocation points in space-time, xi(positions) and t(time), dim(n,2). 
        # - B: tensor whose columns contain the outputs.

        F_a = self.Fa_m*torch.sin(self.C*A[:, 1:2]/self.Tf)
        # F_a = self.Fa_m*torch.sin(np.pi*A[:, 1:2])

        # Functions at the model output
        W = B[:,0:1] # Output of W
        Psi = B[:,1:2] # Output of Psi

        # First to second order derivatives of W respect to A
        W_x = torch.autograd.grad(W, A, grad_outputs=torch.ones_like(W), allow_unused=True, create_graph=True)[0][:,0:1]
        W_xx = torch.autograd.grad(W_x, A, grad_outputs=torch.ones_like(W_x), allow_unused=True, create_graph=True)[0][:,0:1]

        # First and second order derivatives of W respect to T
        W_t = torch.autograd.grad(W, A, grad_outputs=torch.ones_like(W), allow_unused=True, create_graph=True)[0][:,1:2]
        W_tt = torch.autograd.grad(W_t, A, grad_outputs=torch.ones_like(W_t), allow_unused=True, create_graph=True)[0][:,1:2]

        # First to second order derivatives of Psi respect to A
        Psi_x = torch.autograd.grad(Psi, A, grad_outputs=torch.ones_like(Psi), allow_unused=True, create_graph=True)[0][:,0:1]
        Psi_xx = torch.autograd.grad(Psi_x, A, grad_outputs=torch.ones_like(Psi_x), allow_unused=True, create_graph=True)[0][:,0:1]        

        # First to second order derivatives of Psi respect to T
        Psi_t = torch.autograd.grad(Psi, A, grad_outputs=torch.ones_like(Psi), allow_unused=True, create_graph=True)[0][:,1:2]
        Psi_tt = torch.autograd.grad(Psi_t, A, grad_outputs=torch.ones_like(Psi_t), allow_unused=True, create_graph=True)[0][:,1:2]  

        
        gaussian = self._gaussianFunction(A, self.x_i, self.sigma_i)

        # Residuals / Differential equations of the model.
        '''
        Channels of the Gaussian calculation tensor.
        0: concentrented force.
        1: concentrented mass M_1. 
        2: linear elasticity K_1.
        3: torsional elasticity K_t1.
        '''

        if self.Fa_p != 0:
            eq_1 = -Psi_x + W_xx - self.gamma_bs*self.R02*self.K_1*gaussian[2]*W - self.gamma_bs*self.R02*(1 + self.M_1* gaussian[1])*W_tt + self.gamma_bs*self.R02*F_a*gaussian[0]
            # eq_1 = -Psi_x + W_xx - self.gamma_bs*self.R02*W_tt + self.gamma_bs*self.R02*F_a*gaussian[0]
        else:
            eq_1 = -Psi_x + (1 + self.gamma_bs*self.R02*self.P_av)*W_xx - self.gamma_bs*self.R02*self.K_1*gaussian[2] - self.gamma_bs*self.R02*(1 + self.M_1* gaussian[1])*W_tt + self.gamma_bs*self.R02*F_a
            # eq_1 = -Psi_x + W_xx - self.gamma_bs*self.R02*W_tt + self.gamma_bs*self.R02*F_a

        # eq_1 = -Psi_x + W_xx - self.gamma_bs*self.R02*self.K_1*gaussian[2] - self.gamma_bs*self.R02*(1 + self.M_1* gaussian[1])*W_tt + self.gamma_bs*self.R02*F_a
        eq_2 = W_x + self.gamma_bs*self.R02*Psi_xx - (1 + self.gamma_bs*self.R02*self.K_t1*gaussian[3])*Psi - self.gamma_bs*self.R02**2*Psi_tt

        return torch.cat([eq_1, eq_2], dim=1)

    # Calculation of the lower boundary equations, as residuals of the PINN model
    def bclResidue(self, A, B):
        '''
        Inputs:
        - A: Collocation points in the lower boundary conditions, dim(n,2). 
        - B: tensor whose columns contain the outputs Y (displacements) and Psi (rotations), dim(n,2).
        '''

        # Functions at the model output
        W = B[:,0:1] # Output of Y at the lower boundary condition of the domain
        Psi = B[:,1:2] # Output of Psi at the lower boundary condition of the domain

        if self.K_L == 0.0 and self.K_tL == 0.0:
            eq_1 = W
            eq_2 = Psi
        
        elif self.K_L != 0.0 and self.K_tR == 0.0:
            # First order derivatives of W respect to A
            W_x = torch.autograd.grad(W, A, grad_outputs=torch.ones_like(W), allow_unused=True, create_graph=True)[0][:,0:1]
            eq_1 = W - (W_x + Psi)/(self.gamma_bs*self.R02*self.K_L)
            eq_2 = Psi

        elif self.K_L == 0.0 and self.K_tL != 0.0:
            # First to second order derivatives of Psi respect to A
            Psi_x = torch.autograd.grad(Psi, A, grad_outputs=torch.ones_like(Psi), allow_unused=True, create_graph=True)[0][:,0:1]
            eq_1 = W
            eq_2 = Psi - Psi_x/self.K_tL

        elif self.K_L != 0.0 and self.K_tL != 0.0:
            # First order derivatives of W respect to A
            W_x = torch.autograd.grad(W, A, grad_outputs=torch.ones_like(W), allow_unused=True, create_graph=True)[0][:,0:1]

            # First to second order derivatives of Psi respect to A
            Psi_x = torch.autograd.grad(Psi, A, grad_outputs=torch.ones_like(Psi), allow_unused=True, create_graph=True)[0][:,0:1]

            eq_1 = W - (W_x + Psi)/(self.gamma_bs*self.R02*self.K_L)
            eq_2 = Psi - Psi_x/self.K_tL

        return torch.cat([eq_1, eq_2], dim=1)

    # Calculation of the upper boundary equations, as residuals of the PINN model
    def bcuResidue(self, A, B):
        '''
        Inputs:
        - A: Collocation points in the lower boundary conditions, dim(n,2). 
        - B: tensor whose columns contain the outputs Y (displacements) and Psi (rotations), dim(n,2).
        '''
           
        # Functions at the model output
        W = B[:,0:1] # Output of Y at the lower boundary condition of the domain
        Psi = B[:,1:2] # Output of Psi at the lower boundary condition of the domain

        if self.K_R == 0 and self.K_tR == 0:
            eq_1 = W
            eq_2 = Psi
        
        elif self.K_R != 0 and self.K_tR == 0:
            # First order derivatives of W respect to A
            W_x = torch.autograd.grad(W, A, grad_outputs=torch.ones_like(W), allow_unused=True, create_graph=True)[0][:,0:1]
            eq_1 = W + (W_x + Psi)/(self.gamma_bs*self.R02*self.K_R)
            eq_2 = Psi

        elif self.K_R == 0 and self.K_tR != 0:
            # First to second order derivatives of Psi respect to A
            Psi_x = torch.autograd.grad(Psi, A, grad_outputs=torch.ones_like(Psi), allow_unused=True, create_graph=True)[0][:,0:1]
            eq_1 = W
            eq_2 = Psi + Psi_x/self.K_tR

        elif self.K_R != 0 and self.K_tR != 0:
            # First order derivatives of W respect to A
            W_x = torch.autograd.grad(W, A, grad_outputs=torch.ones_like(W), allow_unused=True, create_graph=True)[0][:,0:1]

            # First to second order derivatives of Psi respect to A
            Psi_x = torch.autograd.grad(Psi, A, grad_outputs=torch.ones_like(Psi), allow_unused=True, create_graph=True)[0][:,0:1]

            eq_1 = W + (W_x + Psi)/(self.gamma_bs*self.R02*self.K_R)
            eq_2 = Psi + Psi_x/self.K_tR

        return torch.cat([eq_1, eq_2], dim=1)

    # Calculation of initial conditions (T=0)
    def icResidue(self, A, B):
        '''
        Inputs:
        - A: Collocation points in the lower boundary conditions, dim(n,2). 
        - B: tensor whose columns contain the outputs Y (displacements) and Psi (rotations), dim(n,2).
        '''
    
        # Functions at the model output
        W = B[:,0:1] # Output of Y at the lower boundary condition of the domain
        Psi = B[:,1:2] # Output of Psi at the lower boundary condition of the domain

        # First order derivative of W respect to T
        W_t = torch.autograd.grad(W, A, grad_outputs=torch.ones_like(W), allow_unused=True, create_graph=True)[0][:,1:2]

        # First order derivative of W respect to T
        Psi_t = torch.autograd.grad(Psi, A, grad_outputs=torch.ones_like(W), allow_unused=True, create_graph=True)[0][:,1:2]

        return torch.cat([W, W_t, Psi, Psi_t], dim=1)
    
    # Calculation of the Gaussian function at each dimensionless point object located on the beam
    def _gaussianFunction(self, X, x_pos, sigma_pos):
        '''
        Dimensionless points:
        - Concentrented mass (M_1).
        - Elasticidades lineales (K_1).
        - Elasticidades torcionales (K_t1).
        Inputs:
        - data: tensor dim(m,n) of collocation points on the beam.
        - x_pos: list specifying the location of each element (mean), retrieved from the data JSON.
        - sigma_pos: list specifying the variance of each point element, retrieved from the data JSON.
        '''
        
        # Conversion of data lists to tensors. Their length defines the number of channels
        x_pos_tensor = torch.tensor(x_pos).reshape(len(x_pos),-1,1).to(self.device)
        sigma_pos_tensor = torch.tensor(sigma_pos).reshape(len(sigma_pos),-1,1).to(self.device)

        return torch.exp((-(X[:,0:1] - x_pos_tensor)**2)/(2*sigma_pos_tensor**2))   

class Fit_inverse:
    '''
    Class to execute training processes and metrics
    Input:
    - param_model: JSON file containing dictionaries that group the parameters needed to work with the PINN model.
    '''
    def __init__(self, param_data, param_model):
        self.device = self._device()
        self.model = PINN_inverse(param_data, param_model).to(self.device)
        self.loss_train_dict = {}
        self.coeficients_values_dict = {}

    # Device assignment.
    @staticmethod
    def _device():       
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        # Printing devices, hardware, and CUDA version (if applicable)
        if device.type == 'cuda':
            print("".join(f'{d[0]}: {d[1]}\n' for d in [
                ('Device',device),
                ('Cluster',torch.cuda.get_device_name()),
                ('CUDA',torch.version.cuda),
            ]))
        else:
            print(f'Dispositivo: {device}')

        return device  

    # Xavier uniform weight initialization, gain 1
    def init_xavier_un(self):
        def init_weights(m):
            if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
                g = nn.init.calculate_gain('tanh')
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                m.bias.data.fill_(0)
        self.model.apply(init_weights)
    
    # Train the network and log the metrics
    def train(self, epochs, datasets, loss_fn, optimizer, save_model=False, model_path=None, metric_path=None, coeficients_path=None):
        '''
        Inputs
        - epochs: number of training iterations.
        - type_et: type of model to solve "euler" o "timoshenko".
        - datasets: dictionary with datasets of domain collocation points, boundary conditions, and initial conditions.
        - loss_fn: algorithm for calculating the loss function.
        - optimizer: optimizer algorithm.
        - save_model: 
        - model_path:
        - metric_path:
        '''

        # Initialization of execution time counting
        t0 = datetime.now()

        # Initialization of loss record lists
        loss_train = []
        loss_train_dom = []
        loss_train_bcl = []
        loss_train_bcu = []
        loss_train_ic = []
        loss_train_data = []
        # alpha_values = []

        global loss_list
        loss_list = []

        train_keys = ['Overall', 'PDE', 'BCL', 'BCU', 'IC', 'Data', 'alpha']

        # Network training process
        tags = list(datasets.keys())
        def closure():
            
            optimizer.zero_grad()
            loss_list.clear()
            for tag in tags:
                A = (datasets[tag].I).to(self.device)
                D = (datasets[tag].O).to(self.device)
                if tag == 'PDE':
                    A.requires_grad = True
                    B = self.model(A)
                    D_hat = self.model.governingEquationResidue(A, B)
                    loss = loss_fn(D_hat, D) # original
                    # loss = 0.1*(loss_fn(D_hat[:, 0], D[:, 0]) + loss_fn(D_hat[:, 1], D[:, 1])) # timoshenko
                    loss_list.append(loss)
                elif tag == 'BoundaryConditionsLower':
                    A.requires_grad = True
                    B = self.model(A)
                    D_hat = self.model.bclResidue(A, B)
                    loss = loss_fn(D_hat, D) # original
                    # loss = loss_fn(D_hat[:,0], D[:,0]) + loss_fn(D_hat[:,1], D[:,1]) # timoshenko
                    loss_list.append(loss)
                elif tag == 'BoundaryConditionsUpper':
                    A.requires_grad = True
                    B = self.model(A)
                    D_hat = self.model.bcuResidue(A, B)
                    loss = loss_fn(D_hat, D) # original
                    # loss = loss_fn(D_hat[:,0], D[:,0]) + loss_fn(D_hat[:,1], D[:,1]) # timoshenko
                    loss_list.append(loss)
                elif tag == 'InitialCondition':
                    A.requires_grad = True
                    B = self.model(A)
                    D_hat = self.model.icResidue(A, B)
                    loss = loss_fn(D_hat, D) # original
                    # loss = loss_fn(D_hat[:,0], D[:,0]) + loss_fn(D_hat[:,1], D[:,1]) + loss_fn(D_hat[:,2], D[:,2]) + loss_fn(D_hat[:,3], D[:,3]) # timoshenko
                    loss_list.append(loss)
                elif tag == 'LabelledData':
                    A.require_grad = True
                    B = self.model(A)
                    loss = loss_fn(B, D)
                    loss_list.append(loss)
            
            # Calculation of total loss, weight updates in the network
            total_loss = sum(loss_list)
            total_loss.backward()

            return total_loss

        for epoch in range(epochs):
            optimizer.step(closure)
        
            # Updating loss record lists
            # loss_train.append(sum(loss_list).item())
            loss_list.insert(0, sum(loss_list))        
            loss_train.append(loss_list[0].item())
            loss_train_dom.append(loss_list[1].item())
            loss_train_bcl.append(loss_list[2].item())
            loss_train_bcu.append(loss_list[3].item())
            loss_train_ic.append(loss_list[4].item())
            loss_train_data.append(loss_list[5].item())
            
            # Updating alpha record list
            # alpha_values.append(self.model.alpha_inv_input.item())
            # loss_list.append(self.model.alpha_inv_input.item())

            # Updating alpha as a parameter network
            # self.model.UnknownCoeficient = self.model.alpha_inv_input

            if epoch % 100 == 0:
                log_dict = dict(zip(train_keys, loss_list))
                aux = ''.join(f'{key}: {value:.5e}, ' for key, value in log_dict.items())
                optimizer_str = type(optimizer).__name__
                print(f'##TRAIN## {optimizer_str} - Epoch: {epoch}, ' + aux)

        elapsed_time_train = datetime.now() - t0
        print('\n Training time: ', elapsed_time_train.seconds, '[s]')    

        # Dictionary with recorded losses: total, domain, BC (boundary conditions), IC (initial conditions) and data for estimate the unknown coeficients
        self.loss_train_dict.update({'Overall Loss': loss_train,
                                     'PDE Loss': loss_train_dom,
                                     'BCL loss': loss_train_bcl,
                                     'BCU Loss': loss_train_bcu,
                                     'IC Loss': loss_train_ic,
                                     'DATA Loss': loss_train_data                                                               
                                        })
        
        # Dictionary with recorded the unknown coeficients
        # self.coeficients_values_dict.update({'Alpha values': alpha_values})
        
        # Saving loss records and trained model
        if save_model:
            # self.saveModel(model_path, metric_path, coeficients_path)
            self.saveModel(model_path, metric_path)
    
    # Printing the network architecture
    def structureView(self):
        print('Model structure:')
        for param_tensor in self.model.state_dict():
            print(param_tensor, '\t', self.model.state_dict()[param_tensor].size())

    # Saving losses and the trained model
    # def saveModel(self, model_path, metric_path, coeficients_path):
    def saveModel(self, model_path, metric_path):

        # Dictionary with recorded losses
        torch.save(self.loss_train_dict, metric_path)

        # Dictionary with recorded unknown coeficients
        # torch.save(self.coeficients_values_dict, coeficients_path)
        
        # Model
        torch.save(self.model.state_dict(), model_path)
        # torch.save(self.model.state_dict(), './src/state_model/trained_model.pt')
    
    # Predictions
    def predict(self, A, PATH):
        # A: tensor with positions and times as input data, dim(n,2)

        self.model.load_state_dict(torch.load(PATH, weights_only=True))
        self.model.eval()
        B = self.model(A).detach()
        return B