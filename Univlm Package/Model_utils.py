import os
from UniVLM.Models import Paligemma2B, Qwen2_5_Instruct, Marigold
from UniVLM.Helper import HelperGuy

class ModelGuy:
    def __init__(self):
        """
        Initializes the ModelGuy class, which manages model loading, task execution, 
        and model resource management. It supports multiple models and their respective tasks.

        This class uses HelperGuy to handle input parsing, file checking, and model cleanup.

        Attributes:
            helper_guy (HelperGuy): An instance of HelperGuy to assist with task execution.
            model_classes (dict): A dictionary mapping model names to their corresponding model classes.
            model_tasks (dict): A dictionary mapping model names to the tasks that the model can perform.
            model_instance (object): The currently loaded model instance.
            model_name (str): The name of the currently loaded model.
        """
        self.helper_guy = HelperGuy()

        # Dictionary mapping model names to their respective classes
        self.model_classes = {
            "paligemma2b": Paligemma2B,
            "qwen2_5_instruct": Qwen2_5_Instruct,
            "marigold": Marigold #added Marigold
        }

        # Dictionary mapping model names to their supported tasks
        self.model_tasks = {
            "paligemma2b": ["generate_caption", "answer_question", "batch_generate_caption", "batch_qa"],
            "qwen2_5_instruct": ["generate_text", "answer_instruction"],  # Add Qwen tasks
            "marigold": ["depth_maps", "create_normals"]
        }

        self.model_instance = None
        self.model_name = None

    def load_model(self, model_name):
        """
        Loads the specified model into memory and prepares it for task execution.

        This method checks if the specified model name is supported, then initializes the corresponding model 
        class and calls its specific load function to load the model.

        Parameters:
            model_name (str): The name of the model to load, e.g., "paligemma2b" or "qwen2_5_instruct".
        
        Raises:
            ValueError: If the provided model name is not supported.
        """
        model_name = model_name.lower()
        if model_name in self.model_classes:
            self.model_instance = self.model_classes[model_name]()
            self.model_instance.load_model()  # Call the model's specific load function
            self.model_name = model_name
        else:
            raise ValueError(f"Model '{model_name}' not supported.")

    def run_task(self, task_name, *args, **kwargs):
        """
        Executes the specified task on the currently loaded model.

        This method checks if a model is loaded, verifies if the requested task is supported by the loaded model, 
        and dynamically calls the corresponding method on the model instance.

        Parameters:
            task_name (str): The name of the task to execute, e.g., "generate_caption", "batch_qa".
            *args, **kwargs: Additional arguments required for the task.
        
        Returns:
            object: The result of executing the task.
        
        Raises:
            RuntimeError: If no model is loaded.
            ValueError: If the requested task is not supported by the loaded model.
            NotImplementedError: If the task is not implemented in the model class.
        """
        if self.model_name is None or self.model_instance is None:
            raise RuntimeError("No model is loaded. Use 'load_model' first.")
        
        # Check if the task is supported for the loaded model
        if task_name not in self.model_tasks.get(self.model_name, []):
            raise ValueError(f"Task '{task_name}' is not supported by model '{self.model_name}'.")
        
        # Dynamically call the corresponding function for the task
        task_function = getattr(self.model_instance, task_name, None)
        if task_function is None or not callable(task_function):
            raise NotImplementedError(f"The task '{task_name}' is not implemented in the model class.")
        
        # Call the task function with arguments
        return task_function(*args, **kwargs)

    def process_json_input(self, json_input):
        """
        Processes the JSON input, extracts task details, performs necessary validations,
        loads the specified model, and executes the requested task.

        This function integrates the HelperGuy to handle the parsing of the JSON input, 
        checks file formats, and ensures that the required parameters are provided.

        Parameters:
            json_input (dict): The JSON input specifying the model and task details, 
                               including additional parameters such as image folders or questions.
        
        Returns:
            object: The result of the executed task, or None if an error occurred.
        
        Raises:
            ValueError: If any required parameters are missing or invalid.
        """
        try:
            # Step 1: Parse JSON input using HelperGuy
            parsed_input = self.helper_guy.parse_json_input(json_input)
            model_name = parsed_input["model"]
            task_name = parsed_input["task"]

            # Handle image folder and questions for vision tasks
            image_folder = parsed_input.get("image_folder", None)
            questions = parsed_input.get("questions", None)

            if task_name in ["generate_caption", "batch_generate_caption"]:
                if image_folder:
                    image_files = os.listdir(image_folder)
                    for img_file in image_files:
                        img_path = os.path.join(image_folder, img_file)
                        self.helper_guy.check_file_format(img_path)

            elif task_name in ["batch_qa", "answer_question"]:
                if not questions:
                    raise ValueError("Textual questions input is required for this task.")
                for question, file_path in questions.items():
                    self.helper_guy.check_file_format(file_path)

            elif task_name in ["depth_maps", "create_normals"]:
                if image_folder:
                    image_files = os.listdir(image_folder)
                    for img_file in image_files:
                        img_path = os.path.join(image_folder, img_file)
                        self.helper_guy.check_file_format(img_path)

            # Step 2: Load the model
            self.load_model(model_name)

            # Step 3: Run the task
            if task_name in ["batch_qa"]:
                return self.run_task(task_name, image_folder, questions)
            elif task_name in ["generate_text", "answer_instruction"]:
                prompt = parsed_input.get("prompt", None)
                context = parsed_input.get("context", None)
                return self.run_task(task_name, prompt, context)
            elif task_name in ["depth_maps", "create_normals"]:
                return self.run_task(task_name, image_folder)
            else:
                return self.run_task(task_name, **parsed_input)

        except Exception as e:
            print(f"Error processing JSON input: {e}")
            return None

    def refresh_model(self):
        """
        Refreshes the model by releasing its resources and preparing it for garbage collection.

        This method calls the `refresh_model` function from HelperGuy to clean up the model instance
        and reset the model-related attributes, effectively freeing up memory.

        Raises:
            Exception: If there is no model currently loaded, a message is printed.
        """
        if self.model_instance:
            self.helper_guy.refresh_model(self.model_instance)
            self.model_instance = None
            self.model_name = None
        else:
            print("No model is currently loaded.")
