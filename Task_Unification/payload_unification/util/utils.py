#import os
import subprocess
#from typing import Optional, Union, Any, Dict, List
import torch
from PIL import Image
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel




def switch_conda_env(env_name: str):
    """Switches the Anaconda environment to the specified one."""
    try:
        command = f"conda activate {env_name}"
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print(f"Successfully activated environment: {env_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error activating environment {env_name}: {e}")

def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """Loads an image from the specified path and converts it to a torch.Tensor."""
    image = Image.open(image_path).convert("RGB")
    transform = torch.nn.Sequential(
        torch.nn.functional.interpolate(size=(256, 256), mode="bilinear"),
        torch.nn.functional.to_tensor()
    )
    image_tensor = transform(image)
    return image_tensor