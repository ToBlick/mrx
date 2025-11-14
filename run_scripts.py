import os
import subprocess
import time

def run_all_scripts(directory):
    # Get all Python files in the directory
    python_files = [f for f in os.listdir(directory) if f.endswith(".py")]
    
    for file in python_files:
        file_path = os.path.join(directory, file)
        t1 = time.time()
        print(f"Running {file_path}...")
        result = subprocess.run(
            ["python", file_path],
            env={**os.environ, "MPLBACKEND": "Agg"},  # Set a non-interactive backend
            capture_output=True,
            text=True
        )
        # Run the script
        # result = subprocess.run(["python", file_path], capture_output=True, text=True)
        
        # Print the script output
        print(f"Output of {file}:")
        print(result.stdout)
        if result.stderr:
            print(f"Errors in {file}:")
            print(result.stderr)
        t2 = time.time()
        print('Total run time = ', t2 - t1, ' s')

# Specify the directory containing the scripts
scripts_directory = "scripts/tutorials"
run_all_scripts(scripts_directory)

scripts_directory = "scripts/interactive/"
run_all_scripts(scripts_directory)
