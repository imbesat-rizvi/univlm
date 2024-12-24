

# def depth_pro(payload: Payload) -> int:
#     # Load model and preprocessing transform
#     switch_conda_env("depth_pro")
#     model, transform = depth_pro.create_model_and_transforms()
#     model.eval()

#     # Load and preprocess an image.
#     image = transform(payload.image)

#     # Run inference.
#     prediction = model.infer(image, f_px=payload.f_px)
#     depth = prediction["depth"]  # Depth in [m].
#     return depth

from models.payload import Payload
from load_image_as_tensor import load_image_as_tensor

def depth_pro(payload: Payload) -> int:
    # Load model and preprocessing transform
    #switch_conda_env("depth_pro")
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    # Load and preprocess an image.
    image_path = payload.image_path # assuming payload is corret and someone has transformed the image to tensor
    image=load_image_as_tensor("D:\!!realtest shit\images\DSC_0898.NEF")
    # Run inference.
    prediction = model.infer(image, **payload.kwargs)
    depth = prediction["depth"]  # Depth in [m].
    return depth