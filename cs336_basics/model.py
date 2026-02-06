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

def SiLU(x):
  return x * torch.sigmoid(x)

class SwiGLU(Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
      super().__init__()
      self.d_model = d_model
      self.d_ff = d_ff
      self.w1 = Parameter(torch.nn.init.trunc_normal_(torch.empty(self.d_ff, self.d_model)))
      self.w2 = Parameter(torch.nn.init.trunc_normal_(torch.empty(self.d_model, self.d_ff)))
      self.w3 = Parameter(torch.nn.init.trunc_normal_(torch.empty(self.d_ff, self.d_model)))

    def forward(self, x):
      # print ()
      # print (x.shape)
      # print (self.w1.shape)
      # print (self.w2.shape)
      # FFN(x) = SwiGLU(x, W1 , W2 , W3 ) = W2 @ (SiLU(W1 @ x) * W3 @ x),
      W1X = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
      W3X = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
      hidden = SiLU(W1X) * W3X
      OUT = einsum(hidden, self.w2, "... d_ff, d_model d_ff -> ... d_model")
      return OUT

class RotaryPositionalEmbedding(Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
      super().__init__()
      self.theta = theta
      self.d_k = d_k
      self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
      # print ()
      # print (self.theta, self.d_k, self.max_seq_len)
      # print (x.shape, token_positions.shape)

      # x.shape --> [4, 12, 64]
      # You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
      # token_positions --> [0,1,2,3,4,5,6,7,8,9,10,11] --> actually just assumed x is in order
      
      # I am pretty sure token_positions is for optimization
      # But I think we can simply do R * x

      '''
      cos = torch.cos(torch.Tensor([self.theta]))
      sin = torch.sin(torch.Tensor([self.theta])) 
      R = torch.concat((cos, -sin, sin, cos)).reshape(2, 2)
      Z = torch.zeros((x.shape[-1], x.shape[-1]))
      for z in range(0, x.shape[-1], 2):
        Z[ z:z+2 , z:z+2 ] = R
      OUT = einsum(x, Z, "... d_model, XXX d_model -> ... XXX")
      '''

      # oh our problem is obvious 
      # we arnt doing theta right.
      # its supposed to be a function of i and k.
      # so i appears to be a function of the sequence length ... which we would not be implementing correctly.

      seq_len = x.shape[-2]
      Z = torch.zeros((seq_len, self.d_k, self.d_k))
      for i in range(0, seq_len):
        for k in range(0, self.d_k, 2):
          theta = i / self.theta ** (k / self.d_k)
          Z[i, k+0, k+0] = torch.cos(torch.Tensor([theta]))
          Z[i, k+0, k+1] = -torch.sin(torch.Tensor([theta]))
          Z[i, k+1, k+0] = torch.sin(torch.Tensor([theta]))
          Z[i, k+1, k+1] = torch.cos(torch.Tensor([theta]))
      shape = list(x.shape[0:-1]) + [1] + [x.shape[-1]]
      OUT = torch.sum(Z * x.reshape(shape), axis=-1)
      return OUT

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of scaled dot product attention.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """

    d_k = Q.shape[-1]
    # why is it QTK?
    QTK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / sqrt(d_k)
    QTK[~mask] = -torch.inf
    QTK = torch.softmax(QTK, axis=-1)
    OUT = einsum(QTK, V, "... queries keys, ... keys d_k -> ... queries d_k")
    return OUT

class MultiheadAttention(Module):
    """MultiheadAttention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # So the reason why num_heads isnt used in Q,K,V size is because we divide model by num_heads
        # - Linear Layers (Q, K, V): Three projections (Query, Key, Value) each with (d_{model} * d_{model}) parameters.
        # - Heads (h): Dividing (d_{model}) into (h) heads (where (d_{k}=d_{model}h)) does not change the total parameters.
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Float[Tensor, " ... sequence_length d_model"]):
        """
        Args:
            in_features (Float[Tensor, "... sequence_length d_model"]): Tensor to run your implementation on.

        Returns:
            Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of batched multi-headed attention.
        """
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        #print (q.shape)
        #print (k.shape)
        #print (v.shape)

        batch, seq, _ = q.shape
        q = q.reshape(batch, seq, self.num_heads, self.d_model // self.num_heads).transpose(2,1)
        k = k.reshape(batch, seq, self.num_heads, self.d_model // self.num_heads).transpose(2,1)
        v = v.reshape(batch, seq, self.num_heads, self.d_model // self.num_heads).transpose(2,1)

        # I still dont get how this shape works.
        # QTK[~mask] = -torch.inf
        # - but QTK and mask dimensions dont match ... so what is going on there?

        mask = torch.ones((batch, self.num_heads, seq, seq), dtype=bool)
        mask.tril_()

        # print ()
        # print (q.shape)
        # print (k.shape)
        # print (v.shape)
        # print (mask.shape)
        # mask = torch.triu(mask)

        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(2,1).reshape(batch, seq, self.d_model)
        out = self.output_proj(out)

        return out

class RoPEMultiheadAttention(Module):
    """RotaryPositionalEmbedding MultiheadAttention"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float,
        max_seq_len: int,
    ):
        """
        Args:
            d_model (int): Dimensionality of the feedforward input and output.
            num_heads (int): Number of heads to use in multi-headed attention.
            theta (float): RoPE parameter.
            max_seq_len (int): Maximum sequence length to pre-cache.
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = RotaryPositionalEmbedding(d_k=self.d_k, theta=theta, max_seq_len=max_seq_len)

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"],
    ) -> Float[Tensor, " ... sequence_length d_model"]:

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch, seq, _ = q.shape
        q = q.reshape(batch, seq, self.num_heads, self.d_model // self.num_heads).transpose(2,1)
        # TODO: what is the point of passing token positions?
        q = self.rope(q, token_positions)
        k = k.reshape(batch, seq, self.num_heads, self.d_model // self.num_heads).transpose(2,1)
        # TODO: what is the point of passing token positions?
        k = self.rope(k, token_positions)
        v = v.reshape(batch, seq, self.num_heads, self.d_model // self.num_heads).transpose(2,1)

        mask = torch.ones((batch, self.num_heads, seq, seq), dtype=bool)
        mask.tril_()

        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(2,1).reshape(batch, seq, self.d_model)
        out = self.output_proj(out)

        return out

