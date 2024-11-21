import torch
from torch import nn, Tensor
import torch.nn.functional as F

class PoolAttention(nn.Module):
  def __init__(self, d_model:int, bias:bool=True, device='cpu', dtype=torch.float32) -> None:
    kwargs = {
      'device': device,
      'dtype': dtype
    }
    super().__init__()
    self.d_model = d_model
    self.linear = nn.Linear(d_model, 1, bias, **kwargs)

  def __call__(self, x:Tensor) -> Tensor:
    return super().__call__(x)

  def forward(self, x:Tensor) -> Tensor:
    # * Validate Inputs
    if not isinstance(x, Tensor):
      raise ValueError("'x' must to be a Tensor")
    if x.dim() not in (3, 2):
      text = f"'x' must have shape of (batch, length, d_model) or (length, d_model) but have {tuple(x.shape)}"
      raise ValueError(text)
    if x.dim() == 2:
      x = x.unsqueeze(0)
    if x.shape[-1] != self.d_model:
      raise ValueError(f"'x' {tuple(x.shape)} d_model bust me '{self.d_model}'")

    # * Calculate
    x_l = F.softmax(self.linear(x).moveaxis(-1, 1), dim=-1)
    return (x_l @ x).squeeze(dim=1)

if __name__ == '__main__':
  pool = PoolAttention(128, False)

  x = torch.randn(1, 300, 128)

  z = pool(x)

  print('x', x.shape)
  print('z', z.shape)
