import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location=device))
model.to(device)  # Move model to GPU
model.eval()

raw_img = cv2.imread('D:/tester/input.jpg')
raw_img = torch.from_numpy(raw_img).to(device)  # Move input data to GPU

depth = model.infer_image(raw_img)  # HxW raw depth map

print(depth.cpu().numpy())  # Move the result back to CPU for printing

import matplotlib.pyplot as plt
import numpy as np
# Normalize the depth values for better visualization
depth_normalized = (depth - torch.min(depth)) / (torch.max(depth) - torch.min(depth))
depth_normalized = depth_normalized.cpu().numpy()  # Move the normalized depth back to CPU for visualization