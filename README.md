# Univlm Model Framework

## Description
The Univlm Model Framework provides a flexible and extensible system for loading, processing, and performing inference across various AI models. It aims to simplify interaction with multiple model types and platforms, offering a unified pipeline for Vision-Language Models (VLM). This makes it easier to integrate visual and linguistic information for a range of tasks, such as image captioning, visual question answering, and more.

Our framework supports a variety of models, including those from Hugging Face (HF), VLLM, and other models not natively supported on either platform. This flexibility allows you to easily load and use models from diverse sources, whether they are available on HF, VLLM, or custom-built models that don’t fit into standard frameworks.

## Prerequisites
- We strongly recommend using Conda for a virtual environment. See the [Conda Installation Guide](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).
- OS: Linux

## Installation
A step-by-step guide on how to install the software.

### 1. Install using pip
```bash
pip install univlm
```

### 2. Install external files (one-time setup)
```bash
univlm-install
```

## Quick Start
Refer to the documentation for an overview of library [Doc](https://web-documentation-for-univlm.readthedocs.io/en/latest).

Examples:- 


```python
from univlm.Model import unify

#VLLM Example
prompts = ["Hello, my name is", "what is the capital of United States"]
y = unify("facebook/opt-125m")
y.load()
payload = {"text": prompts, "pixel_values": None}
output = y.inference(payload)
print(output)

# Depth Estimation with Apple DepthPro (Exclusive Model)
y = unify("AppledepthPro")
y.load()
y.Proccessor()
image_path = "input.jpg"
output = y.inference(image_path)
print("Depth map generated:", output)

# Vision-Language Question Answering with BLIP (Vision-Language Model)
from PIL import Image
import requests
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
listy = [raw_image, raw_image]
payload = {"pixel_values": listy, "text": ["how many dogs?", "color of dog"]}
y = unify("Salesforce/blip-vqa-base")
y.load()
y.Proccessor()
output = y.inference(payload)
print(output)

# Image Segmentation with SAM (Vision Model)
img_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
payload = {"pixel_values": image, "text": None}
y = unify("facebook/sam-vit-base", Image_processor=True)
y.load()
y.Proccessor()
output = y.inference(payload)
print(output)

# Sentiment Analysis with BERT (Text Model)
y = unify("nlptown/bert-base-multilingual-uncased-sentiment", Config_Name="BertForNextSentencePrediction")
y.load()
payload = {"text": "Hello, how are you?", "pixel_values": None}
y.Proccessor()
output = y.inference(payload)
print(output)
```

## Contributing
Contributions will be welcomed once the project is finalized.

## License
This project is licensed under the Apache License, Version 2.0, January 2004.
For more details, see: [Apache License 2.0](http://www.apache.org/licenses/).

## Contact
For any inquiries, reach out via email:
- Aryan Singh: sk.singharyan99@gmail.com
- Ilia Davydov: ilyadavydov03@gmail.com
- Siddhant Tyagi: siddhant.tyagizx@gmail.com

### LinkedIn Profiles

- **Project Mentor**: [Imbesat Rizvi](https://www.linkedin.com/in/imbesat-rizvi/)
- [Aryan Singh](https://www.linkedin.com/in/aryan0singh/)
- [Ilia Davydov](https://www.linkedin.com/in/ilia-davydov-783402297/)
- [Siddhant Tyagi](https://www.linkedin.com/in/tyagisiddhant28/)

