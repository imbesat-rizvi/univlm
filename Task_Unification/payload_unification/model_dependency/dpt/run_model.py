import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
model.eval()

raw_img = cv2.imread('input.jpg')
depth = model.infer_image(raw_img) # HxW raw depth map

print(depth)


import matplotlib.pyplot as plt
import numpy as np
# Normalize the depth values for better visualization
depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
# Display the depth map
plt.imshow(depth_normalized, cmap='inferno')  # 'inferno' is a good colormap for depth
plt.colorbar(label="Depth")
plt.title("Depth Map Visualization")
plt.axis('off')
plt.show()
plt.save()
