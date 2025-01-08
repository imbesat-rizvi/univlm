from UniVLM.Model_utils import ModelGuy
from UniVLM.Helper import HelperGuy

def main():
    
    # Updated JSON input with Qwen test task added
    json_input = {
        "tasks": [
            {
                "model": "paligemma2b",
                "task": "batch_generate_caption",
                "image_folder": "Images",
            },
            {
                "model": "paligemma2b",
                "task": "batch_qa",
                "image_folder": "Images",
                "questions": {
                    "What is the image showing": "Images/photomode_01092022_163425.png",
                    "Which game is the image from": "Images/photomode_06092022_162856.png",
                },
            },
            {
                "model": "qwen2_5_instruct",
                "task": "generate_text",
                "prompt": "What is the significance of artificial intelligence in modern healthcare?"
            }
        ]
    }

    # Instantiate HelperGuy and ModelGuy
    helper = HelperGuy()
    model_guy = ModelGuy()

    # Execute tasks
    results = helper.execute_tasks(json_input, model_guy)

    # Print the results
    for task, result in results.items():
        print(f"Results for task '{task}': {result}")


if __name__ == "__main__":
    main()

