import os

class HelperGuy:
    def __init__(self):
        """
        Initializes the HelperGuy class which provides utility methods
        for processing tasks and handling model resources.
        """
        pass
    
    def parse_json_input(self, json_input):
        """
        Parses the JSON input to extract relevant parameters for task execution.
        
        This function assumes that the input JSON is structured in a way that
        allows the tasks to be accessed via the 'tasks' key, where each task
        contains the model name, the type of task, and additional parameters
        necessary for the task execution (e.g., image folder or questions).

        Parameters:
            json_input (dict): A dictionary representing the JSON input containing
                               task specifications such as model name, task type, and 
                               any other required parameters.
        
        Returns:
            dict: The parsed JSON input for further processing.
        
        Raises:
            ValueError: If there is an issue parsing the JSON input or if it is
                        not structured correctly.
        """
        try:
            # Further validation can be implemented here
            return json_input
        except Exception as e:
            print(f"Error parsing JSON input: {e}")
            raise ValueError("Invalid JSON input.")
    
    def check_file_format(self, file_path):
        """
        Verifies the existence and validity of a file format for images or text.

        This function checks whether the specified file exists and if its
        format matches one of the expected formats (PNG, JPG, JPEG, TXT).

        Parameters:
            file_path (str): The file path to be checked.

        Raises:
            FileNotFoundError: If the specified file does not exist at the given path.
            ValueError: If the file format is not supported (i.e., not an image or text file).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.lower().endswith(('png', 'jpg', 'jpeg', 'txt')):
            raise ValueError(f"Invalid file format: {file_path}. Expected an image or text file.")
    
    def refresh_model(self, model_instance):
        """
        Releases resources used by a loaded model and performs necessary cleanup.
        
        This function assumes that the model instance has a method for resource 
        cleanup, which is called to free up memory or other resources associated
        with the model instance.

        Parameters:
            model_instance (object): The model instance to be cleaned up.
        
        Raises:
            Exception: If the model instance does not have a cleanup method or if
                       there is an issue with releasing resources.
        """
        if model_instance:
            model_instance.cleanup()  # Assuming the model has a cleanup method for releasing resources
    
    def execute_tasks(self, json_input, model_guy):
        """
        Executes multiple tasks based on the provided JSON input and the 
        corresponding model to perform each task.
        
        This function iterates over a list of tasks described in the input JSON,
        validates required parameters, and uses the ModelGuy instance to load
        the corresponding model and run the specified task. Supported tasks include:
        - `batch_generate_caption`: For generating captions from a set of images.
        - `batch_qa`: For performing question answering on images.
        - `generate_text`: For generating text from a given prompt.
        - `answer_instruction`: A task for responding to specific instructions (e.g., for Qwen).
        - `depth_maps`: For generating depth_maps for images.
        - `create normals`: For generating normals for images.

        Parameters:
            json_input (dict): The JSON input containing task specifications.
                               The structure of the input includes a 'tasks' key which contains
                               a list of dictionaries, each representing a task.
            model_guy (ModelGuy): An instance of the ModelGuy class, responsible for managing
                                  model loading and task execution.

        Returns:
            dict: A dictionary where the keys are task names and the values are the results
                  of executing each task.

        Raises:
            ValueError: If any required parameter for a task is missing or if an unsupported
                        task is encountered.
            Exception: If there is an error executing any of the tasks.
        """
        results = {}

        try:
            for task in json_input["tasks"]:
                model_name = task["model"]
                task_name = task["task"]

                # Parse input for each task based on its name
                if task_name == "batch_generate_caption":
                    image_folder = task.get("image_folder")
                    if not image_folder:
                        raise ValueError(f"'image_folder' is required for task '{task_name}'.")

                    # Load model and execute captioning
                    model_guy.load_model(model_name)
                    results[task_name] = model_guy.run_task(task_name, image_folder)

                elif task_name == "batch_qa":
                    image_folder = task.get("image_folder")
                    questions = task.get("questions")
                    if not image_folder or not questions:
                        raise ValueError(f"'image_folder' and 'questions' are required for task '{task_name}'.")

                    # Prepare the question list for the task
                    question_list = list(questions.keys())

                    # Load model and execute Q&A
                    model_guy.load_model(model_name)
                    results[task_name] = model_guy.run_task(task_name, image_folder, question_list)

                elif task_name == "generate_text" or task_name == "answer_instruction":
                    # For Qwen text-based tasks (e.g., text generation)
                    prompt = task.get("prompt")
                    if not prompt:
                        raise ValueError(f"'prompt' is required for task '{task_name}'.")

                    # Load model and execute text generation
                    model_guy.load_model(model_name)
                    results[task_name] = model_guy.run_task(task_name, prompt)

                elif task_name == "depth_maps" or task_name == "create_normals":
                    image_folder = task.get("image_folder")
                    if not image_folder:
                        raise ValueError(f"'image_folder'is required for task '{task_name}'.")

                    # Load model and execute
                    model_guy.load_model(model_name)
                    results[task_name] = model_guy.run_task(task_name, image_folder)

                else:
                    raise ValueError(f"Task '{task_name}' is not supported.")

        except Exception as e:
            print(f"Error executing tasks: {e}")
            raise

        return results
