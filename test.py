from PIL import Image
import time
from src import depth_pro
# import torch
# import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb("input.jpg")
image = transform(image)


# Measure inference time
start_time = time.time()

# Run inference.
prediction = model.infer(image)

end_time = time.time()
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

depth = prediction["depth"]  # Depth in [m].
#focallength_px = prediction["focallength_px"]  # Focal length in pixels.

import matplotlib.pyplot as plt

# Normalize the depth map
depth_np = depth.cpu().numpy()
depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

print(depth_np)
# Visualize the normalized depth map
plt.imshow(depth_normalized, cmap='plasma')
plt.colorbar(label='Normalized Depth')
plt.title('Normalized Depth Map')
plt.show()
print("Hello")

# device = torch.device("cuda")
# print(f"Using device: {device}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"Current device: {torch.cuda.current_device()}")
#     print(f"Device name: {torch.cuda.get_device_name()}")