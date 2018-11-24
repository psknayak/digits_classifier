
# NOTE: You can only use Tensor API of PyTorch
from nnet import activation
import torch
# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    """  
    m,no_labels = outputs.size()
    one_hot = torch.zeros(no_labels,m)
    one_hot[labels,torch.arange(labels.size(0))] = 1
    p = activation.softmax(outputs)
    log_likelihood = -(torch.mul(torch.t(one_hot),torch.log(p)))
    creloss = (1/m)*torch.sum(log_likelihood)
    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    """
    labels = labels.unsqueeze(0).float()
    avg_grads = activation.softmax(outputs) - torch.t(labels)
    return avg_grads

if __name__ == "__main__":
    pass
