
import torch
from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch.nn import Module, Parameter, ModuleList
from numpy import sqrt
from einops import einsum, rearrange
import numpy as np

def cross_entropy(inputs, targets):
  '''
  maxes, _ = inputs.max(dim=-1, keepdim=True)
  shift_features = inputs - maxes
  logsumexp = shift_features.exp().sum(dim=-1, keepdim=True).log()
  return (logsumexp - shift_features.gather(-1, targets.unsqueeze(-1))).mean() 
  '''

  '''
  # Compute the log of the softmax of the inputs.
  log_softmax = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
  # Gather the log probabilities of the target classes.
  log_probs = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1))
  # Compute the negative log likelihood.
  loss = -log_probs.mean()
  return loss
  '''

  inputs = inputs - torch.max(inputs, axis=-1, keepdims=True).values
  # Okay so this is tricky.
  # https://www.youtube.com/watch?v=ILmANxT-12I
  # We are interested in categorical cross entropy loss ... or softmax loss
  # So we should assume softmax has not been applied to inputs yet.
  # And cross entropy loss is defined as: -sum( y_i * log(y_i_pred) )
  # But because y_i is 1 for 1 class, and 0 for the others, its just: -1 * y_i_pred
  '''
  softmax = torch.softmax(inputs, axis=-1)
  y = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[-1])
  loss = torch.mean(torch.sum(y * -torch.log(softmax), axis=-1))
  '''
  # So this passed the first test (check test_nn_utils.py) but failed when they scaled inputs by 1000.
  # The issue has to do with numerical stability.
  # softmax(inputs) gives inf and nan when its scaled by 1000.
  # AI told us this: Alternative: LogSoftmax: For loss calculations, using log(softmax(x)) is more numerically stable than calculating log and softmax separately.

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  log_softmax = torch.nn.functional.log_softmax(inputs, dim=-1)
  y = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[-1]).to(device)
  log_probs = torch.sum(y * -log_softmax, axis=-1)
  loss = torch.mean(log_probs)
  return loss

# Tricky part here is that the grad can be "None" which was throwing us off.
def gradient_clipping(parameters, max_l2_norm):
    l2_norm = 0
    for param in parameters:
        grad = param.grad
        if grad is not None:
            l2_norm += torch.sum(grad ** 2)
    l2_norm = l2_norm ** 0.5
    for param in parameters:
        grad = param.grad
        if grad is not None:
            if l2_norm > max_l2_norm:
                grad *= max_l2_norm / (l2_norm + 1e-6)





