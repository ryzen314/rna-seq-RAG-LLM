import subprocess
import sys
import time
import os
import platform
import json
import re

# --- Global variable to store the found executable path ---
OLLAMA_EXECUTABLE_PATH = None

# --- Helper Function for Ollama Environment ---
def _get_ollama_env():
    """
    Creates a safe environment for running Ollama commands, ensuring
    it can find the models directory, especially within a macOS .app bundle.
    """
    env = os.environ.copy()
    if sys.platform == "darwin":
        try:
            home_dir = os.path.expanduser("~")
            models_path = os.path.join(home_dir, '.ollama', 'models')
            env['OLLAMA_MODELS'] = models_path
        except Exception as e:
            print(f"WARNING: Could not set OLLAMA_MODELS path. Error: {e}")
    return env

# --- Command Execution Helper (Updated) ---
def run_command(command_list, capture=True):
    """
    Runs a command with the correct Ollama environment and executable path.
    """
    global OLLAMA_EXECUTABLE_PATH

    # Use the full path to the executable if we found it, otherwise just use 'ollama'
    executable = OLLAMA_EXECUTABLE_PATH or 'ollama'
    # Reconstruct the command with the full path to the executable
    full_command = [executable] + command_list[1:]

    try:
        ollama_env = _get_ollama_env()
        if capture:
            process = subprocess.run(
                full_command, check=True, capture_output=True, text=True, env=ollama_env
            )
            return process.stdout.strip()
        else:
            process = subprocess.run(
                full_command, check=True, capture_output=True, text=True, env=ollama_env
            )
            return process
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running command '{' '.join(full_command)}': {e}")
        return None

# --- Centralized Ollama Functions ---
def find_current_models_downloaded():
    """
    Runs 'ollama list' and returns a list of the names of installed models.
    """
    output = run_command(['ollama', 'list'])
    if output is None:
        return []
    model_names = []
    lines = output.strip().splitlines()[1:]
    for line in lines:
        if line.strip():
            model_name = line.split()[0]
            model_names.append(model_name)
    return model_names

def pull_model(model_name):
    """
    Pulls a model from Ollama. This is a blocking call intended to be
    run in a separate thread. Returns True on success, False on failure.
    """
    print(f"Attempting to pull model: {model_name}")
    process = run_command(['ollama', 'pull', model_name], capture=False)
    return process is not None

# --- VRAM Detection Functions ---
def get_nvidia_vram():
    """Gets total VRAM from nvidia-smi, returns in GB."""
    try:
        output = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
        vram_mb = int(output)
        return f"{vram_mb / 1024:.2f} GB"
    except (Exception):
        return None

def get_amd_gpu_vram():
    """Gets total VRAM from rocm-smi, returns in GB."""
    try:
        # The command returns JSON data, so we parse it
        output = subprocess.run(['rocm-smi', '--showmeminfo', 'vram', '--json'], capture_output=True, text=True, check=True).stdout.strip()
        data = json.loads(output)
        # Find the 'VRAM Total Memory (B)' value, which is in bytes
        total_bytes = int(data['card0']['VRAM Total Memory (B)'])
        return f"{total_bytes / (1024**3):.2f} GB"
    except (Exception):
        return None


def get_apple_silicon_vram():
    """Gets total unified memory on Apple Silicon, returns in GB."""
    if platform.machine() == 'arm64':
        try:
            output = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True, check=True).stdout.strip()
            mem_bytes = int(output.split(': ')[1])
            return f"{mem_bytes / (1024**3):.2f} GB (Unified)"
        except (Exception):
            return None
    return None

def get_system_vram():
    """
    Detects the user's OS and hardware to determine available VRAM.
    """
    print("Detecting system hardware and VRAM...")
    os_type = sys.platform

    if os_type == "darwin":
        vram = get_apple_silicon_vram()
        return ("Apple Silicon", vram) if vram else ("macOS (Intel)", "Detection not implemented")

    elif os_type == "linux":
        # Check for NVIDIA GPU first
        nvidia_vram = get_nvidia_vram()
        if nvidia_vram:
            return ("NVIDIA GPU", nvidia_vram)

        # If not NVIDIA, check for AMD GPU
        amd_vram = get_amd_gpu_vram()
        if amd_vram:
            return ("AMD GPU", amd_vram)

    # Fallback for other systems or if no GPU is detected
    return "Unknown", "Detection not available"

# --- Ollama Management Functions ---
def check_ollama_installed():
    """
    Checks if the Ollama executable exists, sets its global path if found,
    and returns True/False.
    """
    global OLLAMA_EXECUTABLE_PATH
    print("Checking for Ollama installation...")

    # Start with a list of potential paths to check
    paths_to_check = ['ollama'] # Default case for when 'ollama' is in the PATH

    if sys.platform == "darwin":
        # On macOS, add standard installation paths to the list
        paths_to_check.extend([
            "/usr/local/bin/ollama",
            "/usr/bin/ollama",
            "/Applications/Ollama.app/Contents/Resources/ollama"
        ])
    elif sys.platform == "linux":
        paths_to_check.extend(["/usr/local/bin/ollama", "/usr/bin/ollama", "/bin/ollama"])

    # Try to find a working executable from the list of paths
    for path in paths_to_check:
        try:
            # Use 'ollama --version' as a lightweight check to see if it runs
            subprocess.run([path, '--version'], check=True, capture_output=True)
            # If the command succeeds, we found a working executable
            print(f"Ollama found and working at: {path}")
            OLLAMA_EXECUTABLE_PATH = path
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # This path didn't work, so we try the next one
            continue

    print("Ollama executable not found or is not working in standard paths.")
    return False

def serve_ollama():
    """Starts the Ollama server as a background process."""
    global OLLAMA_EXECUTABLE_PATH

    # Use the full path to the executable if we found it, otherwise just use 'ollama'
    executable = OLLAMA_EXECUTABLE_PATH or 'ollama'
    command = f"{executable} serve"

    try:
        ollama_env = _get_ollama_env()
        if sys.platform == "win32":
            subprocess.Popen(f"start /B {command}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=ollama_env)
        else:
            subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid, env=ollama_env)
    except Exception as e:
        print(f"Failed to start Ollama server: {e}")
        sys.exit(1)
