# NOTE: You can only use Tensor API of PyTorch

import torch



# Extra TODO: Document with proper docstring
def sigmoid(z):
    """Calculates sigmoid values for tensors
    
    Argument- Takes in z as a tensor of some shape

    Returns- sigmoid of the tensor z
    """
    ###CODE STARTS HERE###
    result = torch.sigmoid(z)
    ###CODE ENDS HERE###
    return result

# Extra TODO: Document with proper docstring
def delta_sigmoid(z):
    """Calculates derivative of sigmoid function
    derivative of sigmoid function = sigmoid(z)(1-sigmoid(z))

    Argument- Takes in z as a tensor

    Returns- Computed Gradient of the sigmoid function
    """
    ###CODE STARTS HERE###
    grad_sigmoid = torch.mul(sigmoid(z),1-sigmoid(z))
    ###CODE ENDS HERE###
    return grad_sigmoid

# Extra TODO: Document with proper docstring
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors

        Argument- Takes in a tensor x

        Returns- A tensor equal to stable softmax of the tensor x  of the same shape as x 
    """
    x_exp = torch.exp(x - torch.max(x))
    x_sum = torch.sum(x_exp,dim=1,keepdim=True)
    stable_softmax = x_exp/x_sum
    return stable_softmax

if __name__ == "__main__":
    pass
