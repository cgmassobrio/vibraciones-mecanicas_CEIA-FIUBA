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

    def __init__(self, param_model):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        '''
        Open JSON. Dictionaries:
        - physics: physics model parameters.
        - param_model: architecture parameters and the forward stage
        '''
        with open(param_model, 'r') as f:
            data = json.load(f)
        physics = data['physics']
        model_nn = data['parameters']

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
        self.beta_input = nn.Parameter(torch.tensor(1.0))
        self.gamma_input = nn.Parameter(torch.tensor(1.0))
        self.theta_input = nn.Parameter(torch.tensor(1.0))

        self.UnknownBeta = self.beta_input
        self.UnknownGamma = self.gamma_input
        self.UnknownTheta = self.theta_input

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
    def governingEquationResidue_F1(self, A, B):
        # Inputs:
        # - A: Collocation points in space-time, xi(positions) and t(time), dim(n,2). 
        # - B: tensor whose columns contain the outputs.
        
        # Unknown parameter
        beta = self.UnknownBeta
        gamma = self.UnknownGamma
        theta = self.UnknownTheta
        
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

        eq_1 = Psi_tt - Psi_xx + Psi - W_x - beta
        eq_2 = W_tt - W_xx + Psi_x - gamma + theta*torch.sin(A[:, 0:1])*torch.cos(A[:, 1:2])

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

        return torch.cat([W, Psi], dim=1)

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

        return torch.cat([W, Psi], dim=1)

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

class Fit_inverse:
    '''
    Class to execute training processes and metrics
    Input:
    - param_model: JSON file containing dictionaries that group the parameters needed to work with the PINN model.
    '''
    def __init__(self, param_model):
        self.device = self._device()
        self.model = PINN_inverse(param_model).to(self.device)
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
    def train(self, epochs, datasets, loss_fn, optimizer, save_model=False, model_path=None, model_force_path=None, metric_path=None, coeficients_path=None):
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

        # Inicialization of coeficients
        beta_values = []
        gamma_values = []
        theta_values = []

        global loss_list
        loss_list = []

        train_keys = ['Overall', 'PDE', 'BCL', 'BCU', 'IC', 'Data', 'beta', 'gamma', 'theta']

        # Network training process
        tags = list(datasets.keys())
        def closure():
            
            loss_list.clear()
            for tag in tags:
                A = (datasets[tag].I).to(self.device)
                D = (datasets[tag].O).to(self.device)
                if tag == 'PDE':
                    A.requires_grad = True
                    B = self.model(A)
                    D_hat = self.model.governingEquationResidue_F1(A, B)
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
            optimizer.zero_grad()
            optimizer.step(closure)
        
            # Updating loss record lists
            loss_list.insert(0, sum(loss_list))        
            loss_train.append(loss_list[0].item())
            loss_train_dom.append(loss_list[1].item())
            loss_train_bcl.append(loss_list[2].item())
            loss_train_bcu.append(loss_list[3].item())
            loss_train_ic.append(loss_list[4].item())
            loss_train_data.append(loss_list[5].item())
            
            # Updating coeficients record list
            beta_values.append(self.model.beta_input.item())
            gamma_values.append(self.model.gamma_input.item())
            theta_values.append(self.model.theta_input.item())

            loss_list.append(self.model.beta_input.item())
            loss_list.append(self.model.gamma_input.item())
            loss_list.append(self.model.theta_input.item())

            # Updating the coeficients as a parameter network
            self.UnknownBeta = self.model.beta_input
            self.UnknownGamma = self.model.gamma_input
            self.UnknownTheta = self.model.theta_input

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
        self.coeficients_values_dict.update({'Beta values': beta_values,
                                             'Gamma values': gamma_values,
                                             'Theta values': theta_values
                                             })
        
        # Saving loss records and trained model
        if save_model:
            self.saveModel(model_path, model_force_path, metric_path, coeficients_path)
    
    # Printing the network architecture
    def structureView(self):
        print('Model structure:')
        for param_tensor in self.model.state_dict():
            print(param_tensor, '\t', self.model.state_dict()[param_tensor].size())

    # Saving losses and the trained model
    def saveModel(self, model_path, model_force_path, metric_path, coeficients_path):

        # Dictionary with recorded losses
        torch.save(self.loss_train_dict, metric_path)

        # Dictionary with recorded unknown coeficients
        torch.save(self.coeficients_values_dict, coeficients_path)
        
        # Model
        torch.save(self.model.state_dict(), model_path)
    
    # Predictions
    def predict(self, A, PATH):
        # A: tensor with positions and times as input data, dim(n,2)

        self.model.load_state_dict(torch.load(PATH, weights_only=True))
        self.model.eval()
        B = self.model(A).detach()
        return B

    def predictForce(self, A, PATH):
            # A: tensor with positions and times as input data, dim(n,2)

            self.model_force.load_state_dict(torch.load(PATH, weights_only=True))
            self.model_force.eval()
            B = self.model_force(A).detach()
            return B