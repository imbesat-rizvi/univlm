import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

# Check if GPU is available
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
#model.to(device)  # Move model to GPU
model.eval()

# List of input image paths
input_paths = ['assets/examples/demo01.jpg', 'assets/examples/demo02.jpg', 'assets/examples/demo03.jpg', 'assets/examples/demo04.jpg.jpg', 'assets/examples/demo05.jpg', 'assets/examples/demo06.jpg', 'assets/examples/demo07.jpg', 'assets/examples/demo08.jpg']  # Add more paths as needed
output_image_dir = 'D:/tester/output_images/'
output_text_dir = 'D:/tester/output_text/'

for input_path in input_paths:
    raw_img = cv2.imread(input_path)
    #raw_img = torch.from_numpy(raw_img).to(device)  # Move input data to GPU

    depth = model.infer_image(raw_img)  # HxW raw depth map

    # Normalize the depth values for better visualization
    depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    #depth_normalized = depth_normalized.cpu().numpy()  # Move the normalized depth back to CPU for visualization

    # Save the depth map visualization
    plt.imshow(depth_normalized, cmap='inferno')  # 'inferno' is a good colormap for depth
    plt.colorbar(label="Depth")
    plt.title("Depth Map Visualization")
    plt.axis('off')
    output_image_path = output_image_dir + input_path.split('/')[-1].replace('.jpg', '_depth.png')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    # # Save the depth matrix to a text file
    # depth_matrix = depth.cpu().numpy()  # Move the depth matrix back to CPU
    # output_text_path = output_text_dir + input_path.split('/')[-1].replace('.jpg', '_depth.txt')
    # np.savetxt(output_text_path, depth_matrix, delimiter=',', fmt='%f')

    #print(f"Processed {input_path}, saved depth map to {output_image_path} and depth matrix to {output_text_path}")