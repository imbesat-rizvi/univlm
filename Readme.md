# UniVLM Documentation

A comprehensive package for interacting with different models such as Paligemma2B and Qwen2.5-Instruct AI models, providing streamlined interfaces for vision-language tasks and text generation.

## Table of Contents
- [Paligemma2B](#paligemma2b)
- [Qwen2_5_Instruct](#qwen2_5_instruct)
- [HelperGuy](#helperguy)
- [ModelGuy](#modelguy)

## Paligemma2B

A wrapper class for the Paligemma 2B model, providing vision-language capabilities.

### Constructor

```python
def __init__(model_name="google/paligemma-3b-pt-224")
```
**Parameters:**
- `model_name` (str): The name or path of the model to load
  - Default: "google/paligemma-3b-pt-224"

### Methods

#### load_model()
Loads the Paligemma 2B model and associated processor.

**Raises:**
- `RuntimeError`: If the model or processor cannot be loaded successfully

#### generate_caption(image_path, max_new_tokens=50)
Generates a caption for a given image.

**Parameters:**
- `image_path` (str): The file path of the image to generate a caption for
- `max_new_tokens` (int): Maximum number of tokens to generate in the caption
  - Default: 50

**Returns:**
- `str`: The generated caption for the image

**Raises:**
- `RuntimeError`: If captioning fails due to errors in processing the image or model

#### answer_question(image_path, question, max_new_tokens=50)
Answers a question based on the content of a given image.

**Parameters:**
- `image_path` (str): The file path of the image to answer the question about
- `question` (str): The question to ask about the image
- `max_new_tokens` (int): Maximum number of tokens to generate for the answer
  - Default: 50

**Returns:**
- `str`: The answer to the question based on the image content

**Raises:**
- `RuntimeError`: If visual question answering fails

#### batch_generate_caption(image_dir, max_new_tokens=50)
Generates captions for a batch of images in a directory.

**Parameters:**
- `image_dir` (str): Directory containing the images to caption
- `max_new_tokens` (int): Maximum number of tokens per caption
  - Default: 50

**Returns:**
- `dict`: Mapping of image paths to their generated captions

**Raises:**
- `ValueError`: If number of images doesn't match number of prompts
- `RuntimeError`: If batch captioning fails

#### batch_qa(image_dir, questions, max_new_tokens=50)
Performs batch question answering for multiple images.

**Parameters:**
- `image_dir` (str): Directory containing the images
- `questions` (list of str): List of questions, one per image
- `max_new_tokens` (int): Maximum number of tokens per answer
  - Default: 50

**Returns:**
- `dict`: Mapping of image paths to their respective answers

**Raises:**
- `ValueError`: If number of images doesn't match number of questions
- `RuntimeError`: If batch Q&A fails

## Qwen2_5_Instruct

A wrapper class for the Qwen2.5-0.5B-Instruct model for text generation tasks.

### Constructor

```python
def __init__(model_name="Qwen/Qwen2.5-0.5B-Instruct")
```
**Parameters:**
- `model_name` (str): The name or path of the model to load
  - Default: "Qwen/Qwen2.5-0.5B-Instruct"

### Methods

#### load_model()
Loads the Qwen2.5-0.5B-Instruct model and tokenizer.

**Raises:**
- `RuntimeError`: If the model or tokenizer cannot be loaded successfully

#### generate_text(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9)
Generates text based on a given prompt.

**Parameters:**
- `prompt` (str): The input text or question
- `max_new_tokens` (int): Maximum number of tokens to generate
  - Default: 100
- `temperature` (float): Sampling temperature, higher values increase randomness
  - Default: 0.7
- `top_p` (float): Cumulative probability for nucleus sampling
  - Default: 0.9

**Returns:**
- `str`: The generated text

**Raises:**
- `RuntimeError`: If text generation fails

#### answer_instruction(instruction, context=None, max_new_tokens=100)
Generates a response based on an instruction and optional context.

**Parameters:**
- `instruction` (str): The instruction or task description
- `context` (str, optional): Additional context
- `max_new_tokens` (int): Maximum number of tokens for the response
  - Default: 100

**Returns:**
- `str`: The generated response

**Raises:**
- `RuntimeError`: If instruction answering fails

## HelperGuy

Utility class providing helper methods for task processing and resource management.

### Methods

#### parse_json_input(json_input)
Parses JSON input for task execution.

**Parameters:**
- `json_input` (dict): Dictionary containing task specifications
  - Must include: model name, task type, and task-specific parameters

**Returns:**
- `dict`: Parsed JSON input for processing

**Raises:**
- `ValueError`: If JSON input is invalid or improperly structured

#### check_file_format(file_path)
Verifies file existence and format validity.

**Parameters:**
- `file_path` (str): Path to the file to check

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If file format is unsupported

#### execute_tasks(json_input, model_guy)
Executes multiple tasks based on JSON input.

**Parameters:**
- `json_input` (dict): Task specifications
  - Must include: 'tasks' list with model and task details
- `model_guy` (ModelGuy): Instance of ModelGuy for task execution

**Returns:**
- `dict`: Results of executing each task

**Raises:**
- `ValueError`: If required parameters are missing
- `Exception`: If task execution fails

## ModelGuy

Manager class for model loading, task execution, and resource management.

### Constructor
Initializes with supported models and tasks:
- Supported models: paligemma2b, qwen2_5_instruct
- Supported tasks: 
  - Paligemma2B: generate_caption, answer_question, batch_generate_caption, batch_qa
  - Qwen2.5-Instruct: generate_text, answer_instruction

### Methods

#### load_model(model_name)
Loads a specified model into memory.

**Parameters:**
- `model_name` (str): Name of the model to load (e.g., "paligemma2b", "qwen2_5_instruct")

**Raises:**
- `ValueError`: If model name is not supported

#### run_task(task_name, *args, **kwargs)
Executes a specified task on the loaded model.

**Parameters:**
- `task_name` (str): Name of the task to execute
- `*args`, `**kwargs`: Additional task-specific arguments

**Returns:**
- Result of the executed task

**Raises:**
- `RuntimeError`: If no model is loaded
- `ValueError`: If task is not supported
- `NotImplementedError`: If task is not implemented

#### process_json_input(json_input)
Processes JSON input and executes requested tasks.

**Parameters:**
- `json_input` (dict): JSON input specifying model and task details
```json
{
    "model": "model_name",
    "task": "task_name",
    "image_folder": "path/to/images",  // For vision tasks
    "questions": {                     // For QA tasks
        "question1": "image1.jpg",
        "question2": "image2.jpg"
    },
    "prompt": "text_prompt",          // For text generation
    "context": "optional_context"     // For instruction tasks
}
```

**Returns:**
- Task execution results or None if error occurs

**Raises:**
- `ValueError`: If required parameters are missing

#### refresh_model()
Releases model resources and prepares for garbage collection.

**Raises:**
- `Exception`: Prints message if no model is loaded

## Example Usage

```python
# Initialize ModelGuy
model_guy = ModelGuy()

# Example JSON input for image captioning
json_input = {
    "model": "paligemma2b",
    "task": "batch_generate_caption",
    "image_folder": "path/to/images"
}

# Process the task
results = model_guy.process_json_input(json_input)

# Clean up resources
model_guy.refresh_model()
```

## Error Handling

The package implements comprehensive error handling:
- Validates input parameters and file formats
- Checks model and task compatibility
- Provides detailed error messages
- Implements proper resource cleanup

## Resource Management

Automatic resource management features:
- Model cleanup after task completion
- Memory management for batch processing
- Proper release of model resources
- Garbage collection support
