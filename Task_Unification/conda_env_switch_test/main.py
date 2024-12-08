import subprocess

def run_project(project_number):
    if project_number == 1:
        subprocess.run(["conda", "run", "-n", "project1", "python", "project1.py"])
    elif project_number == 2:
        subprocess.run(["conda", "run", "-n", "project2", "python", "project2.py"])
    else:
        print("Invalid project number. Please enter 1 or 2.")

if __name__ == "__main__":
    n = int(input("Enter the number of queries: "))
    for _ in range(n):
        query = input("Enter your query (e.g., 'run 1'): ")
        if query.startswith("run "):
            project_number = int(query.split()[1])
            run_project(project_number)
        else:
            print("Invalid query format. Please use 'run 1' or 'run 2'.")