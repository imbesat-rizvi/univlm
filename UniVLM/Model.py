from .Model_utils import HFModelSearcher,HFProcessorSearcher, reference_table 
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq, AutoModelForSeq2SeqLM, AutoModelForMaskedLM
from vllm import LLM # type: ignore
from PIL import Image
from torchvision import transforms
import torch
import requests,io

class Athena:
    def __init__(self, model_name, Feature_extractor, Image_processor):
        self.model = None
        self.model_type = None  
        self.Processor = None  
        self.model_name = model_name
        self.Feature_extractor = Feature_extractor
        self.Image_processor =  Image_processor
        self.map = None

    def load(self):
        """Determine model type loads it."""

        try: 
            self.model = LLM(model=self.model_name,
                             gpu_memory_utilization=0.9,
                             max_model_len=2048)
            self.model_type = "VLLM"
            print("VLLM model Loaded")
            return "Loaded"  
        except Exception as e:
            print(f"Not supported on VLLM: {e}")

        # Try loading as an HF model
        try:
            placeholder = HFModelSearcher()
            results = placeholder.search(self.model_name)

            if not results:
                raise ValueError("No HF model found.")

            if len(results) > 1:
                print("Multiple use cases found for the same backbone model")
                print(results)
                index = int(input("Write the index of the model you want to use: ")) 
                placeholder2 = results[index][0]  # Assuming it's a tuple/list
                print(reference_table[placeholder2])
                self.model = reference_table[placeholder2]
                
            else:
                placeholder2 = results[0][0]
                self.model = reference_table[placeholder2]
            
            self.map = self.model
            self.model = self.model.from_pretrained(self.model_name)
            self.model_type = "HF"
            print("HF model Loaded")
            return "Loaded"  
        except Exception as e:
            print(f"Not supported on HF: {e}")

        # Try loading as an exclusive model
        try:
            model_class = reference_table[self.model_name]
            self.model = model_class()
            self.model.env_setup()
            self.model.load_model()
            self.model_type = "Exclusive"
            print("Exclusive model Loaded")
            return "Loaded"  
        except Exception as e:
            print(f"Not supported by Athena as of this moment: {e}")

        return "Failed to Load"  # Return failure if all methods fail
       
    def Proccessor(self):
        """
        Determines the appropriate processor (Tokenizer or Processor) for the model
        Args:
            model_name: Name of the model to process
        Returns:
            str: Type of processor selected ('Processor' or 'Tokenizer')
        """

        if self.model_type == "VLLM":
            pass
        elif self.model_type == "HF":
            Placeholder = HFProcessorSearcher()
            self.Processor, temp = Placeholder.search(self.model_name,self.Feature_extractor,self.Image_processor)
            self.Processor = self.Processor.from_pretrained(self.model_name)
        elif self.model_type == "Exclusive":
            pass
        else: 
            raise ValueError("Model not loaded")
        
        return "Processor Loaded"

    def inference(self, payload):

        if self.model_type == "Exclusive":
            self.model.processor(payload)
            return self.model.infer()
        elif self.model_type == "HF":
            try: 
                self.Proccessor()
            except:
                raise ValueError("Processor not loaded")
            
            processor = self.Processor.from_pretrained(self.model_name)
            if payload["text"] and payload["pixel_values"] is None:
                inputs = processor(payload["text"],  return_tensors="pt")
            elif payload["text"] and payload["pixel_values"]:
                inputs = processor(images = payload["pixel_values"], text = payload["text"], return_tensors="pt")
            elif payload["pixel_values"] and payload["text"] is None:
                inputs = processor(images = payload["pixel_values"], return_tensors="pt")

            if self.map in [AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForVision2Seq, AutoModelForMaskedLM]:
                outputs = self.model.generate(**inputs)
                if hasattr(outputs, 'sequences'):
                    generated_ids = outputs.sequences[0]
                else:
                    generated_ids = outputs[0]
                    
                response = processor.decode(generated_ids, skip_special_tokens=True)
                return response
            else:
                try:
                    with torch.no_grad():
                        outputs = []
                        outputs.append(self.model(**inputs))
                        return outputs               
                except: 
                    outputs = self.model.generate(**inputs)
                    generated_ids = outputs[0]
                    response = processor.decode(generated_ids, skip_special_tokens=True)
                    return response
            
        else:
            if payload["text"] and payload["pixel_values"] is None:
                outputs = self.model.generate(payload["text"])
                return outputs[0]
            elif payload["text"] and payload["pixel_values"]:
                outputs = self.model.generate(payload["text"], images = payload["pixel_values"])
                return outputs[0]
            elif payload["pixel_values"] and payload["text"] is None:
                outputs = self.model.generate(payload["pixel_values"])
                return outputs[0]

