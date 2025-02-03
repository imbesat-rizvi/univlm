from UniVLM.Model import Yggdrasil  
from PIL import Image
import requests

# Example of model on HF not vllms
print()
print()
y = Yggdrasil("nlptown/bert-base-multilingual-uncased-sentiment", Feature_extractor=False,Image_processor=False,Config_Name = 'BertForNextSentencePrediction')
y.load()
payload = { "text": "Hello, how are you?", "pixel_values": None }
output = y.inference(payload)
print(output)

# Example of VLM 
print()
print()
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
listy = [raw_image,raw_image]
payload = {"pixel_values": listy, "text": ["how many dogs?","color of dog"]}

y = Yggdrasil("Salesforce/blip-vqa-base", Feature_extractor=False, Image_processor=False)
y.load()

output = y.inference(payload)
print(output)

#Example of Image Only task 
print()
print()
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

payload = {"pixel_values": image, "text": None}

y = Yggdrasil("facebook/sam-vit-base", Feature_extractor=False, Image_processor=True)
y.load()
output = y.inference(payload)
print(output)


#VLLM example 
print()
print()
prompts = ["Hello, my name is", "what is the capital of United States"]
y = Yggdrasil("facebook/opt-125m", Feature_extractor=False, Image_processor=True)
y.load()
payload = {"text": prompts, "pixel_values": None}
output = y.inference(payload)
print(output)

# object detection example why not lmao 
print()
print()
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

payload = {"pixel_values": image, "text": None}

y = Yggdrasil("hustvl/yolos-tiny", Feature_extractor=False, Image_processor=True)
y.load()
output = y.inference(payload)
print(output)

# Depth Estimation SIKE 
print()
print()
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

payload = {"pixel_values": image, "text": None}

y = Yggdrasil("LiheYoung/depth-anything-large-hf", Feature_extractor=False, Image_processor=True)
y.load()
output = y.inference(payload)
print(output)
#%%