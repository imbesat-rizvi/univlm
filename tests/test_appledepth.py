import pytest
from unittest.mock import patch, MagicMock, call
import subprocess
import os
from pathlib import Path
import torch
import tempfile

from UniVLM.Model_utils import appledepth  # Single import

@pytest.fixture
def mock_subprocess():
    with patch("subprocess.run") as mock_run, \
         patch("subprocess.check_output") as mock_check_output:
        yield mock_run, mock_check_output

@pytest.fixture
def mock_path_exists():
    with patch("os.path.exists") as mock_exists:
        yield mock_exists

@pytest.fixture
def mock_temp_image():
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        yield tmp.name

@pytest.fixture
def depth_instance():
    return appledepth()

def test_initialization(depth_instance):
    assert depth_instance.model is None
    assert depth_instance.transform is None
    assert depth_instance.image is None
    assert depth_instance.f_px is None

def test_env_setup_git_conda_check(mock_subprocess, depth_instance):
    mock_run, mock_check_output = mock_subprocess
    
    # Simulate missing git
    mock_run.side_effect = [
        subprocess.CalledProcessError(1, "git"),
        None,
        None,
        None,
        None
    ]
    with pytest.raises(SystemExit):
        depth_instance.env_setup()
    assert "Git is not installed" in str(exc.value)

    # Simulate missing conda
    mock_run.side_effect = [
        None,
        subprocess.CalledProcessError(1, "conda"),
        None,
        None,
        None
    ]
    with pytest.raises(SystemExit):
        depth_instance.env_setup()
    assert "Conda is not installed" in str(exc.value)

def test_env_setup_repo_handling(mock_subprocess, mock_path_exists, depth_instance):
    mock_run, _ = mock_subprocess
    mock_path_exists.side_effect = lambda x: x == "ml-depth-pro"
    
    # Test existing repo
    depth_instance.env_setup()
    assert call(["git", "-C", "ml-depth-pro", "pull"]) in mock_run.call_args_list

    # Test new repo clone
    mock_path_exists.return_value = False
    mock_run.reset_mock()
    depth_instance.env_setup()
    assert call(["git", "clone", "https://github.com/apple/ml-depth-pro.git"]) in mock_run.call_args_list

def test_env_setup_conda_handling(mock_subprocess, depth_instance):
    mock_run, mock_check_output = mock_subprocess
    
    # Test environment creation
    mock_check_output.return_value = '{"envs": []}'
    depth_instance.env_setup()
    assert call(["conda", "create", "-n", "depth-pro", "-y", "python=3.9"]) in mock_run.call_args_list

    # Test existing environment
    mock_check_output.return_value = '{"envs": ["/path/to/depth-pro"]}'
    mock_run.reset_mock()
    depth_instance.env_setup()
    assert call(["conda", "create"]) not in mock_run.call_args_list

def test_load_model_success(depth_instance):
    with patch("src.depth_pro.create_model_and_transforms") as mock_create:
        mock_model = MagicMock()
        mock_transform = MagicMock()
        mock_create.return_value = (mock_model, mock_transform)
        
        depth_instance.load_model()
        
        mock_create.assert_called_once()
        mock_model.eval.assert_called_once()
        assert depth_instance.model == mock_model
        assert depth_instance.transform == mock_transform

def test_load_model_failure(depth_instance):
    with patch("src.depth_pro.create_model_and_transforms", side_effect=ImportError):
        with pytest.raises(ImportError):
            depth_instance.load_model()

def test_processor(mock_temp_image, depth_instance):
    with patch("src.depth_pro.load_rgb") as mock_load_rgb:
        mock_image = MagicMock()
        mock_fx = 100.0
        mock_load_rgb.return_value = (mock_image, None, mock_fx)
        
        depth_instance.transform = MagicMock()
        transformed_image = MagicMock()
        depth_instance.transform.return_value = transformed_image
        
        depth_instance.processor(mock_temp_image)
        
        mock_load_rgb.assert_called_with(mock_temp_image)
        depth_instance.transform.assert_called_with(mock_image)
        assert depth_instance.image == transformed_image
        assert depth_instance.f_px == mock_fx

def test_infer_success(depth_instance):
    mock_image = MagicMock()
    mock_fx = 100.0
    depth_instance.image = mock_image
    depth_instance.f_px = mock_fx
    
    mock_model = MagicMock()
    mock_prediction = torch.rand(256, 256)
    mock_model.infer.return_value = mock_prediction
    depth_instance.model = mock_model
    
    result = depth_instance.infer()
    
    mock_model.infer.assert_called_with(mock_image, f_px=mock_fx)
    assert result.shape == (256, 256)

def test_infer_failure(depth_instance):
    with pytest.raises(RuntimeError):
        depth_instance.infer()

def test_conda_env_exists():
    with patch("subprocess.check_output") as mock_check:
        mock_check.return_value = '{"envs": ["/path/to/envs/depth-pro"]}'
        assert appledepth.conda_env_exists("depth-pro") is True
        
        mock_check.return_value = '{"envs": ["/path/to/envs/other-env"]}'
        assert appledepth.conda_env_exists("depth-pro") is False

def test_models_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "ml-depth-pro"
        (repo_dir / "checkpoints").mkdir(parents=True)
        
        # Test model exists
        (repo_dir / "checkpoints" / "depth_pro.pt").touch()
        assert appledepth.models_exist(repo_dir) is True
        
        # Test missing model
        (repo_dir / "checkpoints" / "depth_pro.pt").unlink()
        assert appledepth.models_exist(repo_dir) is False

def test_model_download_handling(mock_subprocess, depth_instance):
    mock_run, _ = mock_subprocess
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "ml-depth-pro"
        script_path = repo_dir / "get_pretrained_models.sh"
        
        # Test model download
        depth_instance.env_setup()
        assert mock_run.call_count == 6  # Verify download commands
        
        # Test existing model
        (repo_dir / "checkpoints" / "depth_pro.pt").touch()
        mock_run.reset_mock()
        depth_instance.env_setup()
        assert "get_pretrained_models.sh" not in str(mock_run.call_args_list)