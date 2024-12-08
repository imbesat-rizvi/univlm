# Conda-env-switch-test

**Here we try and test to run multiple(2 as of now) conda environment based on a user query**


We create two separate Conda environments, project1 and project2, each with its specific dependencies.

1. Create the First Environment:
```bash
conda create -n project1 python=3.10  # Adjust Python version as needed
```
You can check the list of env using the following command, you should see the: **project1               C:\Users\sksin\anaconda3\envs\project1**
```bash
conda env list
```
After that you must activate the conda env using the following command:
```bash
conda activate project1
```
After activation in the terminal you would have a **(project1)** before every line *for eg:* (project2) D:\tester\conda_ene_switch_test>conda deactivate

Install any and all dependencies you want:
```bash
pip install numpy pandas matplotlib  # Add specific dependencies for project 1
```

**Note:** make sure you deactivate your env using conda deactivate before creating the second one
```bash 
conda deactivate
```
2. Create the Second Environment:
Just repeat the steps for 1st env creation
```bash
conda create -n project2 python=3.10
```

```bash
conda activate project2
```

```bash
pip install numpy pandas scikit-learn  # Add specific dependencies for project 2
```
3. Now you enter the conda base environment:
```bash
conda activate base
```
4. Run the main.py script:
```bash
python main.py
```
**NOTE:** If you face issue that after runnig the *"conda activate <env-name>"* command it asks you to run conda init but still nothing happens, just open a terminal with administrator access and run everything from that its the last thing that worked for me(after trying all the other solutions.)