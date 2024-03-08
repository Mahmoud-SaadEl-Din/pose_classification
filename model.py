import torch
import torch.nn as nn

class CustomFeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(CustomFeedforwardNN, self).__init__()

        # Check if hidden_sizes is a list or a single integer
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        # Build the neural network architecture
        layers = []
        in_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        # Add output layer
        layers.append(nn.Linear(in_size, output_size))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)
        self.model.apply(self.init_weights_he)

    def forward(self, x):
        return torch.sigmoid(self.model(x))
    
    def init_weights_he(self, m):
        if isinstance(m, nn.Linear):
            print("here")
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(m.bias)


