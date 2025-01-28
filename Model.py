import json
from Model_utils import updater, Mapper, CONFIG_MAPPING_NAMES, Loader_Lazy, Base
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor

#potentially you will have to import diffuser and then something like image_processor etc for the Processor function on line 101. 

#from vllm import LLM
#from src import depth_pro
import torch

class UnifiedLoader:
    def __init__(self):
        with open('Model_mapping.json', 'r') as f:
            self.model_mappings = json.load(f)
        
        self.model = None
        self.model_type = False
        self.is_vllm = False
        self.model_map = None
        self.pro_token = None 
        self.Proccessor = None
        self.pro_token = None
        self.unsupported_models = False

    def load(self, model_name: str, model_family: str):
        """
        Determine model type and whether it's a VLLM model
        Returns (model_type, is_vllm) tuple
        """
        # if model_family in self.model_mappings["vllm_supported_models"]:
        #     self.model_map = "vllm_supported_models"
        #     self.is_vllm = True
        #     self.model_type = True
            
        for model_key in self.model_mappings["MODEL_FOR_CAUSAL_LM_MAPPING_NAMES"]:
            if model_key.lower() in model_family.lower():
                self.model_map = "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES"
                self.model_type = True
                self.pro_token = "generate"
                
        for model_key in self.model_mappings["MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"]:
            if model_key.lower() in model_family.lower():
                self.model_map = "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"
                self.model_type = True
                self.pro_token = "generate"
        
        for model_key in self.model_mappings["MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES"]:
            if model_key.lower() in model_family.lower():
                self.model_map = "MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES"
                self.model_type = True
                self.pro_token = "forward"
        
        for model_key in self.model_mappings["MODEL_FOR_MASK_GENERATION_MAPPING_NAMES"]:
            if model_key.lower() in model_family.lower():
                self.model_map = "MODEL_FOR_MASK_GENERATION_MAPPING_NAMES"
                self.model_type = True
                self.pro_token = "forward"
        
        for model_key in self.model_mappings["MODEL_EXCEPTIONAL_CLASS"]:
            if model_key.lower() in model_family.lower():
                self.model_map = "MODEL_EXCEPTIONAL_CLASS"
                self.model_type = True
                #self.pro_token = "forward"
                self.unsupported_models = True
        
        if not self.model_type:
            raise ValueError(f"Model {model_name} not found in supported models")

        # if self.unsupported_models:
        #     # exception_calss = Mapper[model_key]
        #     # exception_calss.env_setup()
        #     # self.model = exception_calss.load_model()/
        #     self.model = Mapper[model_key]
        #     self.model.env_setup()
        #     self.model.load_model()
        #     return "Loaded Model"

        if self.unsupported_models:
            model_class = Mapper[model_key]
            self.model = model_class()  # Create an instance of the class
            self.model.env_setup()
            self.model.load_model()
            return "Loaded Model" # return self.model 

        # if self.is_vllm:
        #     print("0")      
        #     self.model = LLM(model=model_name)  
        #     return "Loaded Model"
        try:
            print("HF model")
            Placeholder = Loader_Lazy(CONFIG_MAPPING_NAMES, Mapper[self.model_map])
            class Loader(Base):
                 _model_mapping = Placeholder
            updater(Loader)
            self.model = Loader.from_pretrained(model_name)
            return "Loaded Model"
    
        except Exception as e:
            raise Exception(f"Error loading model {model_name}: {str(e)}")
        
    def Processor(self, model_name ,task):
        """
        Determines the appropriate processor (Tokenizer or Processor) for the model
        Args:
            model_name: Name of the model to process
        Returns:
            str: Type of processor selected ('Processor' or 'Tokenizer')
        """
        #self.pro_token = None

        if(task != "IMAGE_PROCESSING"):
            # debug :print("kuch to hai ji!")
            # Check if model requires a Processor
            if "PROCESSOR_MAPPING_NAMES" in self.model_mappings:
                for model_key in self.model_mappings["PROCESSOR_MAPPING_NAMES"]:
                    if model_key.lower() in model_name.lower():
                        #self.pro_token = "Processor"
                        self.Proccessor = AutoProcessor
                        break
            
            # If no processor found, check for Tokenizer
            if "TOKENIZER_MAPPING_NAMES" in self.model_mappings:
                for model_key in self.model_mappings["TOKENIZER_MAPPING_NAMES"]:
                    if model_key.lower() in model_name.lower():
                        #self.pro_token = "Tokenizer"
                        self.Proccessor = AutoTokenizer
                        break
        else:
            if "IMAGE_PROCESSOR_MAPPING_NAMES" in self.model_mappings:

                for model_key in self.model_mappings["IMAGE_PROCESSOR_MAPPING_NAMES"]:
                    if model_key.lower() in model_name.lower():
                        #self.pro_token = "ImageProcessor"
                        self.Proccessor = AutoImageProcessor
                        break
        
        # If neither found, default to Tokenizer
        if self.pro_token is None:
            print(f"Warning: No specific processor found for {model_name}. Defaulting to Tokenizer.")
            #self.pro_token = "Tokenizer"
            self.Proccessor = AutoTokenizer
        
        return self.pro_token   
 
    def inference(self, payload, model_name, model_family ,task):

        if self.unsupported_models:
            self.model.processor(payload)
            return self.model.infer()
            

        if not self.is_vllm:
            if not self.model:
                raise ValueError("Model not loaded")
            self.Processor(model_family,task)
            if not self.Proccessor:
                raise ValueError("Processor not loaded")
                
            processor = self.Proccessor.from_pretrained(model_name)
            inputs = processor(payload, return_tensors="pt")

            if self.pro_token == "generate":
                outputs = self.model.generate(**inputs)
                # Handle the output tensor properly
                if hasattr(outputs, 'sequences'):
                    generated_ids = outputs.sequences[0]
                else:
                    generated_ids = outputs[0]
                    
                response = processor.decode(generated_ids, skip_special_tokens=True)
                return response
            
            else:
                #outputs = self.model(**inputs) 
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    #predicted_depth = outputs.predicted_depth
                    #masks = processor.image_processor_type.post_process_masks(
                    # outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
                    # )
                    return outputs               
            #outputs = self.model.generate(**inputs)  # Changed from self.model(**inputs)
            
            
        # else:
        #     outputs = self.model.generate("Hello, my name is")
        #     return outputs[0].text



#-----------------talks----------------------
# initially task to unifi all models on hf and othewise and 
# the choice for the user to choose between running model using vLLM or huggingface

# single command for loading all the models=> load
# single command for running all the models=> inference

# currntly pipeline inegrating all the models on huggingface transformers library
# and a blueprint for adding models not supported by huggingface=> depthPro by apple.inc

# remains is integrating models for hf diffuser library => for eg marigold
# asssume we have integrated the above.

#------------------------------------------------------------------------
#next seteps towards completion of the project.

#testing all the different models on huggingface and diffuser library
# for eg all differnt types of models are supported on slurm.=> my job now

#software testing: handel exceptions , wrong inputs etc 

# packaging and publishing 
# documentation

#we can sit together and work on integrating your model i.e.e marigold .
#------------------------------------------------------------------------

#Lets begin with using then seeing what happens in the backend.
#there is a load() function how does this work?
#you can check https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/auto_pipeline.py to see the list of models 
#that are supported by the library diffusers.
# searches from the list of supported models and returns provide the necessay things for Process function. 
#Process function is used to determine the appropriate processor (Tokenizer or Processor) for the model.

# the inference will be called.
# the nlp models use .generate() to get output.
# check protoken and call the model accordingly.

#%%
# y = UnifiedLoader()
# from PIL import Image

# # depth anything large

# y.load("LiheYoung/depth-anything-large-hf", "depth_anything") # loading the model
# image = Image.open("input.jpg").convert("RGB") # loading the image
# x = y.inference(image,"LiheYoung/depth-anything-large-hf", "depth_anything", "IMAGE_PROCESSING")
# print(x)


# #sam
# y.load("facebook/sam-vit-huge", "sam")
# image = Image.open("input.jpg").convert("RGB")
# x = y.inference(image,"facebook/sam-vit-huge", "sam", "IMAGE_PROCESSING")
# print(x)


#Apple Depth Pro
# y.load("appleDepthPro", "appleDepthPro")
# #image = Image.open("input.jpg").convert("RGB")
# image = "input.jpg"
# x = y.inference(image,"appleDepthPro", "appleDepthPro", "IMAGE_PROCESSING")
# print(x)

#y.load("Salesforce/blip-image-captioning-base", "blip")
# image = Image.open("input.jpg").convert("RGB")
# x = y.inference("What is your name","Qwen/Qwen2-0.5B", "Qwen2", "NOT_IMAGE_PROCESSING")
# print(x)