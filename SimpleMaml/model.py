import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidModel(nn.Module):
    def __init__(self, N_points):
        super(SinusoidModel, self).__init__()
        self.linear1 = nn.Linear(N_points, 40)
        self.linear2 = nn.Linear(40,40)
        self.linear3 = nn.Linear(40,N_points)
    
    def forward(self, x):
        out = self.linear3(nn.ReLU(self.linear2(nn.ReLU(self.linear1(x)))))

        return out


# Output without using model
def argforward(weights, x):
    output = F.linear(x,weights[0],weights[1])
    output = F.relu(output)
    output = F.linear(output,weights[2],weights[3])
    output = F.relu(output)
    output = F.linear(output,weights[4],weights[5])

    return output

def clone_weights(model):
    #GNN Weights
    weights=[w.clone() for w in model.parameters()]

    return weights