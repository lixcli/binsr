from torch.autograd import Function
import torch
import torch.nn as nn

def TorchSign():
    class binarize(Function):
        @staticmethod
        def forward(ctx,x):
            ctx.save_for_backward(x)
            return torch.sign(x)
        
        @staticmethod
        def backward(ctx,grad_outputs):
            x = ctx.saved_tensors[0]
            grad_input = (2 - torch.abs(2 * x)).clamp(min=0)
            grad_input = grad_input * grad_outputs.clone()
            return grad_input
    return binarize().apply

class binary_weight(nn.Module):
    """
    binary function for weight
    """
    def __init__(self):
        super(binary_weight,self).__init__()
        self.round = TorchSign()
    
    def forward(self,w):
        # w_mean = w.detach().mean([1,2,3], keepdim=True) # output channels
        # w_var = w.detach().var([1,2,3], keepdim=True)
        scale_factor = w.abs().detach().mean([1,2,3], keepdim=True)
        bw = self.round(w)
        return bw*scale_factor

class binary_activation(nn.Module):
    def __init__(self):
        super(binary_activation,self).__init__()
        self.round = TorchSign()
    
    def forward(self,a):
        # w_mean = w.detach().mean([1,2,3], keepdim=True) # output channels
        # w_var = w.detach().var([1,2,3], keepdim=True)
        ba = self.round(a)
        return ba