import torch
from PIL import Image

def load_image_as_tensor(image_path: str) -> torch.Tensor:
    """Loads an image from the specified path and converts it to a torch.Tensor."""
    image = Image.open(image_path).convert("RGB")
    transform = torch.nn.Sequential(
        torch.nn.functional.interpolate(size=(256, 256), mode="bilinear"),
        torch.nn.functional.to_tensor()
    )
    image_tensor = transform(image)
    return image_tensor