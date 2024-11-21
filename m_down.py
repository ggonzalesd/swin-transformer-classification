import torch
from torch import nn, Tensor
import torch.nn.functional as F

from m_conv import Convolutional

class DownBlock(nn.Module):
  def __init__(self,
    in_ch:int, out_ch:int,
    active_fn=F.silu,
    device='cpu', dtype=torch.float32
  ) -> None:
    factory_kwargs = { 'device': device, 'dtype': dtype }
    super().__init__()

    self.active_fn = active_fn

    self.conv = nn.Sequential(
      Convolutional(in_ch, in_ch, residual=True, active_fn=active_fn, **factory_kwargs),
      Convolutional(in_ch, out_ch, residual=False, active_fn=active_fn, **factory_kwargs),
      nn.MaxPool2d(2),
    )

  def __call__(self, X:Tensor) -> Tensor:
    return super().__call__(X)

  def forward(self, X:Tensor) -> Tensor:
    X = self.conv(X)
    return X
