from Model import UnifiedLoader as univlm
from PIL import Image


model=univlm.load("LiheYoung/depth-anything-large-hf", "depth_anything")
image = Image.open("input.jpg").convert("RGB") # loading the image
x = univlm.inference(image,"LiheYoung/depth-anything-large-hf", "depth_anything", "IMAGE_PROCESSING")
print(x)