
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import numpy as np
import warnings
from scipy.stats import norm
import matplotlib.pyplot as plt
from evaluation_metrics import gaussian_NLL


# -----------------------------------------------------------------------------
#                               FCNN
# -----------------------------------------------------------------------------
class FCNN(nn.Module):
    def __init__(self, input_dim, n_layers, n_neurons, output_dim, tr_layers=0):
        super(FCNN, self).__init__()
        self.layers = nn.ModuleList() # initialize the layers list as an empty list using nn.ModuleList()
        
        self.init_params = [input_dim, n_layers, n_neurons, output_dim, tr_layers]
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.output_dim = output_dim
        self.tr_layers = tr_layers
        
        in_features = input_dim # input dimension for the layer
        
        # define the number of neurons in each layer
        tr_layers = tr_layers+1  # something about the math that comes after is a bit confusing but somehow with this +1 it works
        n = list()
        n.append(input_dim)
        for i in range(tr_layers-1):   # -1 because if not it starts creating the following one
            n.append( input_dim +(n_neurons-input_dim)*(i+1)//tr_layers) 
        for _ in range(n_layers):
            n.append(n_neurons)
        for i in range(tr_layers-1):
            n.append( output_dim +(n_neurons-output_dim)*(tr_layers-i-1)//tr_layers) 
        n.append(output_dim)

        # Hidden layers
        for i in range(len(n) - 1):
            # Add layer
            self.layers.append(nn.Linear(n[i], n[i+1]))
        
    def _initialize_weights(self, seed=None):
            if seed is not None:
                torch.manual_seed(seed)  # Set the seed for reproducibility
                np.random.seed(seed)  # Set the seed for numpy (if needed)
            
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    # Initialize weights and biases
                    nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))  # He initialization
                    if layer.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                        bound = 1 / np.sqrt(fan_in)
                        nn.init.uniform_(layer.bias, -bound, bound)  # Uniform initialization for biases

    def forward(self, x):    # Function to perform forward propagation
        for layer in self.layers[:-1]: # no activation in the last layer
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
    def fit(self, data, 
            epochs=1000, 
            criterion=None, 
            optimizer=None, 
            weight_decay=0.0,
            batch_size=None,
            learning_rate=0.01,
            verbose=True,
            n_prints=20, # if verbose, how many times should plot the loss in the console
            use_gpu=True,
            plot_loss=True
            ):
        
        """Trains the model"""
        
        if optimizer is None:
            optimizer=optim.Adam(self.parameters(), lr=learning_rate)
        
        # initialize batch size for both training and testing datasets
        if batch_size is None:
            batch_size = len(data['x_train'])
        if 'x_test' in data:
            batch_size_test = batch_size
            if len(data['x_test']) < batch_size_test:
                batch_size_test = len(data['x_test'])
            
        # initialize the loss function
        if criterion is None:
            criterion = nn.MSELoss()
            
        if use_gpu is True:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                warnings.warn("GPU not found on the current computing system. Training model on CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
        
        self.train()
        loss_list = []
        loss_test_list = []
        self.to(device)
        for epoch in range(epochs):
            running_loss = 0.0
            for idx in ordered_indices_generator(len(data['x_train']), batch_size):
                X_batch = data['x_train'][idx, :].to(device)
                Y_batch = data['y_train'][idx, :].to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = self.forward(X_batch)
                loss = criterion(outputs, Y_batch)

                # Add L1 regularization
                l1_reg = 0.0
                for param in self.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += weight_decay * l1_reg

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            # Save training loss
            loss_list.append(running_loss)
            
            # Release memory on gpu
            del X_batch
            del Y_batch
            
            # Testing loop
            if (('x_test' in data) and ('y_test' in data)):
                running_test_loss = 0.0
                for idx in ordered_indices_generator(len(data['x_test']), batch_size_test):
                    X_batch = data['x_test'][idx, :].to(device)
                    Y_batch = data['y_test'][idx, :].to(device)
                    
                    with torch.no_grad():
                        # Forward pass + loss
                        outputs = self.forward(X_batch)
                        test_loss = criterion(outputs, Y_batch)
                        
                    # Add L1 regularization
                    test_loss += weight_decay * l1_reg.detach() # was computed previously

                    running_test_loss += test_loss.item()
                
                loss_test_list.append(running_test_loss)

            if (epoch + 1) % (epochs//n_prints) == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Training loss: {running_loss / len(data['x_train'])}")
        self.train_loss = loss_list
        self.test_loss  = loss_test_list
