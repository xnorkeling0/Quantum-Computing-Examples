# PRODHUB LLC - The Quantum Computing Company
Contributing to quantum computing advancement by exploring practical applications. 

# Quantum Computing Examples
Live Collection of Python scripts to get started with Quantum Computing coding

## System Setup
 - system: macOS
 - create virtual environment: `python3 -m venv .venv`
 - activate `.venv` with `source .venv/bin/activate`
 - `pip install --upgrade pip`
 - `pip intall -r requirements.txt`
 - in `~/.bashrc` do `export IBM_QUANTUM_TOKEN="<your_ibm_token_here>"`
 - `source ~/.bashrc` on Linux or `source ~/.zshrc` on MacOS
 - in your script the token is accessed via `token = os.getenv('IBM_QUANTUM_TOKEN')`

 ## Usage
 To run Classical and Quantum Computing models
 In CLI run `python src/quantum_machine_learning/run_models.py` 

 Other examples can be launched by calling them in `src/main.py` and 
running the CLI command `python src/main.py` 
