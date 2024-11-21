import torch
from torch import Tensor, nn
import torch.nn.functional as F

from einops import rearrange

class SwinTransformerEncoder(nn.Module):
  def __init__(self,
    d_model:int, nhead:int, dim_feedforward:int=725,
    bias=False, activation=F.gelu,
    device='cpu', dtype=torch.float32,
  ) -> None:
    kwargs = {
      'device': device,
      'dtype': dtype,
    }
    super().__init__()

    self.d_model = d_model
    self.encoder = nn.TransformerEncoderLayer(
      d_model, nhead, dim_feedforward,
      activation=activation, bias=bias,
      batch_first=True, norm_first=True,
      **kwargs
    )

  def __call__(self, x:Tensor, window_size:int, shift:bool=False) -> Tensor:
    return super().__call__(x, window_size, shift)

  def forward(self, x:Tensor, window_size:int, shift:bool) -> Tensor:
    # * Calculate
    b, h, _, _ = x.shape
    if shift:
      x = torch.roll(x, (-window_size // 2, -window_size // 2), (1, 2))
    x = rearrange(x, 'b (dh p1) (dw p2) c -> (b dh dw) (p1 p2) c', p1=window_size, p2=window_size)

    x:Tensor = self.encoder(x)

    x = rearrange(x, '(b dh dw) (p1 p2) c -> b (dh p1) (dw p2) c', b=b, dh=h//window_size, p1=window_size)
    if shift:
      x = torch.roll(x, (window_size // 2, window_size // 2), (1, 2))
    return x

class SwinTransformer(nn.Module):
  def __init__(self,
    window_size:int, d_model:int, layers:int, nhead:int, dim_feedforward:int=725,
    bias=False, activation=F.gelu,
    device='cpu', dtype=torch.float32,
  ) -> None:
    kwargs = {
      'device': device,
      'dtype': dtype,
    }
    super().__init__()

    self.window_size = window_size
    self.d_model = d_model
    self.layers = nn.ModuleList([
      nn.ModuleList([
        SwinTransformerEncoder(d_model, nhead, dim_feedforward, bias, activation, **kwargs)
        for _ in range(2)
      ])
      for _ in range(layers)
    ])

  def __call__(self, x:Tensor) -> Tensor:
    return super().__call__(x)

  def forward(self, x:Tensor) -> Tensor:
    # * Validate Inputs
    if not isinstance(x, Tensor):
      raise ValueError("'x' must to be a Tensor")
    if x.dim() not in (4, 3):
      text = f"'x' must have shape of (batch, height, width, d_model) or (height, width, d_model) but have {tuple(x.shape)}"
      raise ValueError(text)
    if x.dim() == 3:
      x = x.unsqueeze(0)
    if x.shape[-1] != self.d_model:
      raise ValueError(f"'x' {tuple(x.shape)} d_model bust me '{self.d_model}'")
    b, h, w, _ = x.shape
    if h % self.window_size != 0 or w % self.window_size != 0:
      raise ValueError(f"Width and Height 'x' {tuple(x.shape)} must be divided by window_size({self.window_size})")

    # * Calculate
    for first, second in self.layers:
      first:SwinTransformerEncoder
      second:SwinTransformerEncoder
      x = first(x, self.window_size, False)
      x = second(x, self.window_size, True)
    return x

if __name__ == '__main__':
  swin = SwinTransformer(
    window_size=16,
    d_model=128, nhead=8, layers=12,
    activation=F.silu, bias=True
  )

  x = torch.randn(3, 32, 32, 128)

  z = swin(x)

  print('x', x.shape)
  print('z', z.shape)
