from PIL import Image
import time
from src import depth_pro

def initialize_depth_pro():
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()
    return model

def transform_image(image_path):
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)
    return image, f_px

# # Initialize the model
# model, transform = initialize_depth_pro()

# # Load and preprocess an image.
# image, _, f_px = depth_pro.load_rgb("example.jpg")
# image = transform(image)

model = initialize_depth_pro()
image = "example.jpg"
image, f_px = transform_image(image)

# Measure inference time
start_time = time.time()

# Run inference.
prediction = model.infer(image, f_px=f_px)

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

print(depth)