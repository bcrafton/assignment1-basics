import torch
from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch.nn import Module, Parameter, ModuleList
from numpy import sqrt
from einops import einsum, rearrange
#from cs336_basics.nn_utils import softmax

'''
def Linear(torch.nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype=None):
    self.in_features = in_features
    self.out_features = out_features
    self.device = device
    self.dtype = dtype
    self.weights = torch.

  def forward(self, x):
'''

@torch.no_grad()
def trunc_normal(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -3.0,
    b: float = 3.0,
) -> Tensor:
    """Fill the input Tensor with values drawn from a truncated normal distribution.

    Args:
        tensor: `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value"""

    if std is None:
        std = 1 / sqrt(tensor.size(0))
    tensor.normal_(mean=mean).mul_(std).clamp_(min=a, max=b)
    return tensor


class Linear(Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
      super().__init__()
      self.in_features = in_features
      self.out_features = out_features
      self.weight = torch.empty(in_features, out_features)
      std = 1 / sqrt(self.in_features)
      self.weight = torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
      self.weight = Parameter(self.weight)

    def forward(self, in_features):
      # return einsum(in_features, self.weight, "... d_in, d_out d_in -> ... d_out")
      # return in_features @ self.weight.T
      # print (in_features.shape, self.weight.shape)
      return einsum(in_features, self.weight, "... din, dout din -> ... dout")

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
      super().__init__()
      self.num_embeddings = num_embeddings
      self.embedding_dim = embedding_dim
      self.weight = torch.empty(num_embeddings, embedding_dim)
      std = 1 / sqrt(self.num_embeddings)
      self.weight = torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
      self.weight = Parameter(self.weight)

    def forward(self, token_ids):
      return self.weight[token_ids]

class RMSNorm(Module):
    def __init__(self, d_model, eps, device=None, dtype=None):
      super().__init__()
      self.d_model = d_model
      self.eps = eps
      self.weight = torch.empty(d_model)
      std = 1 / sqrt(self.d_model)
      self.weight = torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)
      self.weight = Parameter(self.weight)

    def forward(self, x):
      assert x.shape[-1] == self.d_model
      RMS = torch.sqrt( self.eps + (1. / self.d_model) * torch.sum(torch.square(x), axis=-1, keepdims=True))
      return x * self.weight / RMS













