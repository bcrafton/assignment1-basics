
import torch
from jaxtyping import Float, Int
import torch
from torch import Tensor
from torch.nn import Module, Parameter, ModuleList
from numpy import sqrt
from einops import einsum, rearrange
import numpy as np

"""
Given a dataset (a 1D numpy array of integers) and a desired batch size and
context length, sample language modeling input sequences and their corresponding
labels from the dataset.

Args:
    dataset (np.array): 1D numpy array of integer token IDs in the dataset.
    batch_size (int): Desired batch size to sample.
    context_length (int): Desired context length of each sampled example.
    device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
        to place the sampled input sequences and labels on.

Returns:
    Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
    is the sampled input sequences, and the second tuple item is the corresponding
    language modeling labels.
"""

def get_batch(dataset, batch_size, context_length, device):
  start = torch.randint(low=0, high=len(dataset)-context_length, size=(batch_size,))
  x = [ dataset[a:a+context_length] for a in start ]
  y = [ dataset[a+1:a+context_length+1] for a in start ]
  x = torch.from_numpy(np.array(x))
  y = torch.from_numpy(np.array(y))
  x.to(torch.device(device))
  y.to(torch.device(device))
  return x, y

"""
Given a serialized checkpoint (path or file-like object), restore the
serialized state to the given model and optimizer.
Return the number of iterations that we previously serialized in
the checkpoint.

Args:
    src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
    model (torch.nn.Module): Restore the state of this model.
    optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
Returns:
    int: the previously-serialized number of iterations.
"""

def load_checkpoint(src, model, optimizer):
    loaded_checkpoint = torch.load(src)
    model.load_state_dict(loaded_checkpoint["model"])
    optimizer.load_state_dict(loaded_checkpoint["optimizer"])
    return loaded_checkpoint["iteration"]

"""
Given a model, optimizer, and an iteration number, serialize them to disk.

Args:
    model (torch.nn.Module): Serialize the state of this model.
    optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
    iteration (int): Serialize this value, which represents the number of training iterations
        we've completed.
    out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
"""

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)





