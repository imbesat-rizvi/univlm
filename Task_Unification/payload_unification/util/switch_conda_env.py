import subprocess

def switch_conda_env(env_name: str):
    """Switches the Anaconda environment to the specified one."""
    try:
        command = f"conda activate {env_name}"
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print(f"Successfully activated environment: {env_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error activating environment {env_name}: {e}")