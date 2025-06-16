import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=37, n_class=3, hidden_dims=[128, 64], dropout=0.5):
        super(MLPClassifier, self).__init__()
        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            # layers.append(nn.Dropout(p=dropout))
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, n_class))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.model(x)

        return logits