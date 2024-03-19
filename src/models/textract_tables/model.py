import torch
import torch.nn as nn
import torch.nn.functional as F

class TableClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_prob=0.5):
        super(TableClassifier, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        self.dropout = nn.Dropout(dropout_prob)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_layers[0])])
        
        for h1, h2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.hidden_layers.append(nn.Linear(h1, h2))
            self.batch_norms.append(nn.BatchNorm1d(h2))
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, x):
        for layer, bn in zip(self.hidden_layers, self.batch_norms):
            x = F.relu(layer(x))
            x = bn(x)
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)


