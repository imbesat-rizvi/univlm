from typing import Optional, Union
from pydantic import BaseModel
import torch
#import torch->kyuki bahar wale mai dala hua hoga

class DPTModel(BaseModel):
    x: torch.Tensor
    f_px: Optional[Union[float, torch.Tensor]] = None
    interpolation_mode: str = "bilinear"