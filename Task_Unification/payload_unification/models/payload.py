from pydantic import BaseModel
from typing import string
from kwargs.dpt import DPTModel
import torch # doubt: this need to be check ki andar bahar dono jagh torch hoga ki nahi

class Payload(BaseModel):
    image_path: string
    instance_dpt: DPTModel # naming has to be better

