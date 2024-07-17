from chemprop import data, featurizers, models, nn
import torch
import torch.nn.functional as F

def build_model():
        
    #Create the network
    mp = nn.BondMessagePassing(depth=3) # Make the gnn
    agg = nn.MeanAggregation() # Aggregation type. Can also do SumAgg. or NormAgg.
    ffn = nn.RegressionFFN() # regression head

    # I haven't experimented with this at all, not sure if it will affect the SSL
    batch_norm = False

    #initialize the model
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, [nn.metrics.MSEMetric])

    return mpnn


def clone_weights(model):
    #GNN Weights
    weights=[w.clone() for w in model.parameters()]
    
    return weights

def message(H, bmg):
    index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
    M_all = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
        0, index_torch, H, reduce="sum", include_self=False
    )[bmg.edge_index[0]]
    M_rev = H[bmg.rev_edge_index]

    return M_all - M_rev

def update(M_t, H_0, weights):
    """Calcualte the updated hidden for each edge"""
    H_t = F.linear(M_t, weights, None)
    H_t = F.relu(H_0 + H_t)

    return H_t

def finalize(M, V, weights, biases):
    H = F.linear(torch.cat((V, M), dim=1), weights, biases)  # V x d_o
    H = F.relu(H)

    return H

# Output without using model
def argforward(weights, bmg):
    H_0 = F.linear(torch.cat([bmg.V[bmg.edge_index[0]],bmg.E],dim=1), weights[0],None)
    H = F.relu(H_0)

    for i in range(3):
        M = message(H,bmg)
        H = update(M,H_0, weights[1])


    index_torch = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
    M = torch.zeros(len(bmg.V), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(
            0, index_torch, H, reduce="sum", include_self=False
        )
    H_v = finalize(M,bmg.V,weights[2], weights[3])
    H = nn.MeanAggregation(H_v, bmg.batch)

    output = F.linear(H,weights[4],weights[5])
    output = F.relu(output)
    output = F.linear(output,weights[6],weights[7])

    return output
