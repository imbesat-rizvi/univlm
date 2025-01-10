import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from UniVLM.Helper import HelperGuy
import pytest
from unittest.mock import Mock

# def test_parse_json_input():
    # helper = HelperGuy()
    
    # # Valid JSON input
    # valid_json = {"tasks": [{"model": "model1", "task": "generate_text", "prompt": "Hello"}]}
    # assert helper.parse_json_input(valid_json) == valid_json

    # # Invalid JSON input
    # with pytest.raises(ValueError):
    #     helper.parse_json_input(None)  # Simulating invalid input

def test_check_file_format(tmp_path):
    helper = HelperGuy()

    # Create a temporary file
    valid_file = tmp_path / "valid_image.png"
    valid_file.touch()

    # Test with valid file
    helper.check_file_format(str(valid_file))

    # Test with unsupported format
    invalid_file = tmp_path / "invalid_file.pdf"
    invalid_file.touch()
    with pytest.raises(ValueError):
        helper.check_file_format(str(invalid_file))

    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        helper.check_file_format("non_existent_file.txt")

def test_refresh_model():
    helper = HelperGuy()
    mock_model = Mock()
    
    # Simulate model cleanup
    helper.refresh_model(mock_model)
    mock_model.cleanup.assert_called_once()

def test_execute_tasks():
    helper = HelperGuy()

    # Mock inputs
    json_input = {
        "tasks": [
            {"model": "model1", "task": "batch_generate_caption", "image_folder": "folder_path"},
            {"model": "model2", "task": "generate_text", "prompt": "Hello World"}
        ]
    }

    # Mock ModelGuy instance
    model_guy = Mock()
    model_guy.run_task.side_effect = ["caption_result", "text_result"]

    # Execute tasks
    results = helper.execute_tasks(json_input, model_guy)

    # Assertions
    assert results["batch_generate_caption"] == "caption_result"
    assert results["generate_text"] == "text_result"
    model_guy.load_model.assert_any_call("model1")
    model_guy.load_model.assert_any_call("model2")
    model_guy.run_task.assert_any_call("batch_generate_caption", "folder_path")
    model_guy.run_task.assert_any_call("generate_text", "Hello World")
