import pytest
import json
import os
import sys
from typing import Dict, Any
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from UniVLM.Models import Paligemma2B, Qwen2_5_Instruct

MODEL_CLASS_MAP = {
    "paligemma2b": Paligemma2B,
    "qwen2_5_instruct": Qwen2_5_Instruct,
}

class SimilarityMatcher:
    def __init__(self):
        # Initialize sentence transformer model for semantic similarity
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate basic string similarity using SequenceMatcher"""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def get_text_similarity(self, expected: str, actual: str) -> float:
        """
        Calculate text similarity using sentence embeddings and cosine similarity
        """
        # Get embeddings for both texts
        embeddings = self.sentence_transformer.encode([expected, actual], convert_to_tensor=True)
        
        # Normalize the embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.mm(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0).t()).item()
        
        return similarity

    def is_similar_enough(self, expected: str, actual: str, task_type: str,
                         threshold: Dict[str, float] = None) -> bool:
        """
        Check if texts are similar enough based on the task type
        
        Args:
            expected: Expected output text
            actual: Actual model output text
            task_type: Type of task ('caption', 'qa', or 'text_generation')
            threshold: Dictionary of thresholds for different task types
        """
        if threshold is None:
            threshold = {
                'caption': 0.6,    # Higher threshold for image captions
                'qa': 0.65,        # Higher for QA as answers should be more precise
                'text_generation': 0.5  # Lower for general text generation
            }
        
        # Convert both texts to strings if they aren't already
        expected = str(expected)
        actual = str(actual)
        
        # For very short outputs (like single word answers), use string similarity
        if len(expected.split()) <= 3 or len(actual.split()) <= 3:
            similarity = self.get_string_similarity(expected, actual)
            return similarity >= threshold[task_type]
        
        # For longer outputs, use semantic similarity
        similarity = self.get_text_similarity(expected, actual)
        return similarity >= threshold[task_type]

def load_model_registry(registry_path: str) -> Dict[str, Any]:
    try:
        with open(registry_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model registry file not found at {registry_path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in model registry file")

class TestModelTasks:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.registry_path = os.path.join(
            os.path.dirname(__file__),
            'model_registry_test.json'
        )
        self.model_registry = load_model_registry(self.registry_path)
        self.similarity_matcher = SimilarityMatcher()
        
    def prepare_task_args(self, task_input: Any) -> tuple:
        """
        Prepare arguments for task execution based on input format.
        
        Args:
            task_input: Input data from test case (can be str, dict, or other types)
            
        Returns:
            tuple: (args, kwargs) where args is a list of positional arguments 
            and kwargs is a dict of keyword arguments
        """
        # Case 1: Dictionary input - contains named parameters
        if isinstance(task_input, dict):
            # Special case for visual QA tasks
            if 'image_path' in task_input and 'question' in task_input:
                return [task_input['image_path'], task_input['question']], {}
            
            # For other visual tasks, pass image_path as positional arg
            if 'image_path' in task_input:
                args = [task_input['image_path']]
                kwargs = {k: v for k, v in task_input.items() if k != 'image_path'}
                return args, kwargs
            
            # For other cases, convert first item to positional arg
            first_key = next(iter(task_input))
            args = [task_input[first_key]]
            kwargs = {k: v for k, v in task_input.items() if k != first_key}
            return args, kwargs
            
        # Case 2: String or simple value input - single positional argument
        else:
            return [task_input], {}
    def compare_outputs(self, expected_output: Any, actual_output: Any, model_name: str, task_name: str) -> bool:
        """
        Compare expected and actual outputs, handling both single outputs and batch outputs.
        
        Args:
            expected_output: Expected output (str or dict)
            actual_output: Actual output (str or dict)
            model_name: Name of the model being tested
            task_name: Name of the task being tested
        """
        # Handle single string outputs (non-batch tasks)
        if isinstance(expected_output, str):
            if not isinstance(actual_output, str):
                print(f"Type mismatch: expected str, got {type(actual_output)}")
                return False
                
            # Clean up outputs
            expected = expected_output.lower().strip()
            actual = (actual_output.lower()
                    .replace("describe this image.", "")
                    .replace("\n", " ")
                    .strip())
            
            # For debugging
            print(f"Comparing single outputs:")
            print(f"  Expected: '{expected}'")
            print(f"  Actual:   '{actual}'")
            
            task_type = self.get_task_type(model_name, task_name)
            return self.similarity_matcher.is_similar_enough(expected, actual, task_type)
        
        # Handle dictionary outputs (batch tasks)
        elif isinstance(expected_output, dict):
            if not isinstance(actual_output, dict):
                print(f"Type mismatch: expected dict, got {type(actual_output)}")
                return False
                
            for image_path, expected_caption in expected_output.items():
                # Normalize paths to handle both forward and backslashes
                normalized_path = os.path.normpath(image_path)
                actual_path = next((k for k in actual_output.keys() 
                                if os.path.normpath(k) == normalized_path), None)
                
                if actual_path is None:
                    print(f"Missing output for image: {normalized_path}")
                    return False
                
                actual_caption = actual_output[actual_path]
                
                # Clean up captions
                expected_caption = expected_caption.lower().strip()
                actual_caption = (actual_caption.lower()
                                .replace("describe this image.", "")
                                .replace("\n", " ")
                                .strip())
                
                # For debugging
                print(f"Comparing captions for {normalized_path}:")
                print(f"  Expected: '{expected_caption}'")
                print(f"  Actual:   '{actual_caption}'")
                
                task_type = self.get_task_type(model_name, task_name)
                if not self.similarity_matcher.is_similar_enough(
                    expected_caption, actual_caption, task_type):
                    print(f"Similarity check failed for {normalized_path}")
                    return False
                    
            return True
        
        else:
            print(f"Unsupported output type: {type(expected_output)}")
            return False

    def get_task_type(self, model_name: str, task_name: str) -> str:
        """Determine the type of task for similarity threshold selection"""
        if model_name == "paligemma2b":
            if "caption" in task_name:
                return "caption"
            elif "question" in task_name:
                return "qa"
        return "text_generation"

    @pytest.mark.parametrize("model_name", MODEL_CLASS_MAP.keys())
    def test_model_tasks(self, model_name: str):
        if model_name not in self.model_registry:
            pytest.skip(f"No test configuration found for model {model_name}")
            
        model_config = self.model_registry[model_name]
        
        model_class = MODEL_CLASS_MAP[model_name]
        model = model_class()
        
        try:
            model.load_model()
        except Exception as e:
            pytest.fail(f"Failed to load model {model_name}: {str(e)}")
        
        for task_name, test_cases in model_config['tasks'].items():
            for test_case in test_cases:
                args, kwargs = self.prepare_task_args(test_case['input'])
                print(args, kwargs)
                expected_output = test_case['expected_output']
                
                try:
                    result = model.run_task(task_name, *args, **kwargs)
                    
                    assert self.compare_outputs(
                        expected_output, result, model_name, task_name
                    ), f"Output similarity check failed for task {task_name}"
                        
                except Exception as e:
                    pytest.fail(
                        f"Task {task_name} failed for model {model_name}: {str(e)}"
                    )

if __name__ == "__main__":
    pytest.main([__file__])
