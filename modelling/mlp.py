import torch
import torch.nn as nn


activation_functions = {
    'ReLU': nn.ReLU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh,
    'LeakyReLU': nn.LeakyReLU
}

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_fn="ReLU"):
        """
        Args:
            input_size (int): Number of input features.
            hidden_sizes (list of int): List containing the number of units for each hidden layer.
            output_size (int): Number of output units.
            activation (nn.Module): Activation function class (default: nn.ReLU).
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.act_fn = act_fn

        layers = []
        prev_size = input_size
        
        #assert act_fn in activation_functions.keys(), f"Must specifiy one of the following activations {",".join(activation_functions.keys())}."

        activation = activation_functions[act_fn]
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation())  # Apply activation function
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))  # Output layer
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        torch.save({
            'class_name': self.__class__.__name__,
            'kwargs': {"input_size": self.input_size,
                       "hidden_sizes": self.hidden_sizes ,
                       "output_size": self.output_size,
                       "act_fn": self.act_fn},
                       'state_dict': self.state_dict()
                       }, path)