from torch import nn
import torch.nn.functional as F


class WorldsBestModel(nn.Module):
    def __init__(self, n_input, n_output, hidden_layers, p=0.2):
        super().__init__()

        # Hidden layer
        if not hidden_layers:
            raise ValueError(f'Expected hidden layers but got {hidden_layers}')

        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, hidden_layers[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], n_output)

        # Dropout
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        if int(x.shape[1]) != self.hidden_layers[0].in_features:
            raise ValueError(f'Expected input to have shape [n, {self.hidden_layers[0].in_features}] got {x.shape}')

        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)

        # Output
        x = F.log_softmax(self.output(x), dim=1)
        return x
