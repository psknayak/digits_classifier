
# NOTE: You can only use Tensor API of PyTorch

import torch
from activation import softmax
# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    """  
    m = labels.size(0)
    p = softmax(outputs)
    log_likelihood = -(torch.mul(labels,torch.log(p)))
    creloss = (1/m)*torch.sum(log_likelihood)
    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    """
    difference = labels - outputs;
    avg_grads = torch.mul(difference,softmax(outputs))
    return avg_grads

if __name__ == "__main__":
    pass

    x=torch.arange(1.,10.).view(3,3)
    y=torch.arange(20.,29.).view(3,3)
    print(cross_entropy_loss(x,y))
    print(delta_cross_entropy_softmax(x,y))