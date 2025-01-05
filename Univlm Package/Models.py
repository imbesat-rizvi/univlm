from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import os

class Paligemma2B:
    def __init__(self, model_name="google/paligemma-3b-pt-224"):
        """
        Initializes the Paligemma 2B model with the given model name.

        Args:
            model_name (str): The name or path of the model to load (default is "google/paligemma-3b-pt-224").
        """
        self.model_name = model_name
        self.model = None
        self.processor = None

    def load_model(self):
        """
        Loads the Paligemma 2B model and associated processor from the specified model name.

        Raises:
            RuntimeError: If the model or processor cannot be loaded successfully.
        """
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            print(f"Paligemma 2B model '{self.model_name}' loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading Paligemma 2B model: {e}")

    def generate_caption(self, image_path, max_new_tokens=50):
        """
        Generates a caption for a given image.

        Args:
            image_path (str): The file path of the image to generate a caption for.
            max_new_tokens (int): The maximum number of tokens to generate in the caption (default is 50).

        Returns:
            str: The generated caption for the image.

        Raises:
            RuntimeError: If captioning fails due to errors in processing the image or model.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            raise RuntimeError(f"Image captioning failed: {e}")

    def answer_question(self, image_path, question, max_new_tokens=50):
        """
        Answers a question based on the content of a given image.

        Args:
            image_path (str): The file path of the image to answer the question about.
            question (str): The question to ask about the image.
            max_new_tokens (int): The maximum number of tokens to generate for the answer (default is 50).

        Returns:
            str: The answer to the question based on the image content.

        Raises:
            RuntimeError: If visual question answering fails due to errors in processing the image or model.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, text=question, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            return answer
        except Exception as e:
            raise RuntimeError(f"Visual question answering failed: {e}")

    def batch_generate_caption(self, image_dir, max_new_tokens=50):
        """
        Generates captions for a batch of images located in the specified directory.

        Args:
            image_dir (str): The directory containing the images to caption.
            max_new_tokens (int): The maximum number of tokens to generate for each caption (default is 50).

        Returns:
            dict: A dictionary mapping image paths to generated captions.

        Raises:
            ValueError: If the number of images does not match the number of prompts.
            RuntimeError: If batch captioning fails due to issues with image processing or the model.
        """
        try:
            image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
            prompts = ["Describe this image."] * len(images)

            if len(images) != len(prompts):
                raise ValueError(f"Number of images ({len(images)}) does not match number of prompts ({len(prompts)})")

            inputs = self.processor(images=images, text=prompts, return_tensors="pt", padding=True)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            captions = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
            return dict(zip(image_paths, captions))

        except Exception as e:
            raise RuntimeError(f"Batch image captioning failed: {e}")

    def batch_qa(self, image_dir, questions, max_new_tokens=50):
        """
        Performs batch question answering for a set of images located in a directory, where each image
        corresponds to a respective question.

        Args:
            image_dir (str): The directory containing the images to perform Q&A on.
            questions (list of str): A list of questions, one for each image.
            max_new_tokens (int): The maximum number of tokens to generate for each answer (default is 50).

        Returns:
            dict: A dictionary mapping image paths to answers for each respective question.

        Raises:
            ValueError: If the number of images does not match the number of questions.
            RuntimeError: If batch Q&A fails due to issues with image processing or the model.
        """
        try:
            image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('png', 'jpg', 'jpeg'))]
            images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

            if len(questions) != len(images):
                raise ValueError(f"Number of images ({len(images)}) does not match number of questions ({len(questions)})")

            inputs = self.processor(images=images, text=questions, return_tensors="pt", padding=True)
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            answers = [self.processor.decode(output, skip_special_tokens=True) for output in outputs]
            return dict(zip(image_paths, answers))

        except Exception as e:
            raise RuntimeError(f"Batch image Q&A failed: {e}")

class Qwen2_5_Instruct:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initializes the Qwen2.5-0.5B-Instruct model with the specified model name.

        Args:
            model_name (str): The name or path of the model to load (default is "Qwen/Qwen2.5-0.5B-Instruct").
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Loads the Qwen2.5-0.5B-Instruct model and tokenizer from the specified model name.

        Raises:
            RuntimeError: If the model or tokenizer cannot be loaded successfully.
        """
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print(f"Qwen2.5-0.5B-Instruct model '{self.model_name}' loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Error loading Qwen2.5-0.5B-Instruct model: {e}")

    def generate_text(self, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
        """
        Generates text based on the given prompt using the Qwen2.5-0.5B-Instruct model.

        Args:
            prompt (str): The input text or question.
            max_new_tokens (int): The maximum number of tokens to generate (default is 100).
            temperature (float): Sampling temperature (default is 0.7). Higher values lead to more randomness.
            top_p (float): Cumulative probability for nucleus sampling (default is 0.9).

        Returns:
            str: The generated text.

        Raises:
            RuntimeError: If text generation fails due to model or tokenization issues.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text
        except Exception as e:
            raise RuntimeError(f"Text generation failed: {e}")

    def answer_instruction(self, instruction, context=None, max_new_tokens=100):
        """
        Generates a response based on an instruction and optional context.

        Args:
            instruction (str): The instruction or task description.
            context (str): Optional additional context.
            max_new_tokens (int): The maximum number of tokens to generate for the response (default is 100).

        Returns:
            str: The generated response.

        Raises:
            RuntimeError: If instruction answering fails due to model or tokenization issues.
        """
        try:
            input_text = f"Context: {context}\nInstruction: {instruction}" if context else f"Instruction: {instruction}"
            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        except Exception as e:
            raise RuntimeError(f"Instruction answering failed: {e}")
