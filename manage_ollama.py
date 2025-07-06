import subprocess
import sys
import time
import os
import platform
import json
import re

# --- Command Execution Helper ---
def run_command(command_list):
    """Runs a command and returns its output, suppressing errors."""
    try:
        process = subprocess.run(
            command_list,
            check=True,
            capture_output=True,
            text=True
        )
        return process.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Return None if the command fails or is not found
        return None

# --- VRAM Detection Functions ---
def get_nvidia_vram():
    """Gets total VRAM from nvidia-smi, returns in GB."""
    output = run_command(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
    if output:
        try:
            vram_mb = int(output)
            return f"{vram_mb / 1024:.2f} GB"
        except (ValueError, IndexError):
            return None
    return None

def get_amd_vram():
    """Gets total VRAM from rocm-smi, returns in GB."""
    output = run_command(['rocm-smi', '--showmeminfo', 'vram', '--json'])
    if output:
        try:
            data = json.loads(output)
            # Find the VRAM total, which is usually in bytes
            vram_bytes = int(data['card0']['VRAM Total Memory (B)'])
            return f"{vram_bytes / (1024**3):.2f} GB"
        except (json.JSONDecodeError, KeyError, IndexError):
            return None
    return None

def get_apple_silicon_vram():
    """Gets total unified memory on Apple Silicon, returns in GB."""
    # On Apple Silicon, VRAM is the total unified system memory.
    if platform.machine() == 'arm64':
        output = run_command(['sysctl', 'hw.memsize'])
        if output:
            try:
                # Output is like: "hw.memsize: 17179869184"
                mem_bytes = int(output.split(': ')[1])
                return f"{mem_bytes / (1024**3):.2f} GB (Unified)"
            except (IndexError, ValueError):
                return None
    return None

def get_system_vram():
    """
    Detects the user's OS and hardware to determine available VRAM.
    """
    print("Detecting system hardware and VRAM...")
    os_type = sys.platform

    if os_type == "darwin": # macOS
        vram = get_apple_silicon_vram()
        if vram:
            return "Apple Silicon", vram
        else:
            # Fallback for Intel Macs if needed, though less common now
            return "macOS (Intel)", "VRAM detection not implemented for Intel Macs."

    elif os_type == "linux" or os_type == "win32":
        # Try NVIDIA first, as it's very common
        vram = get_nvidia_vram()
        if vram:
            return "NVIDIA", vram
        
        # If NVIDIA fails, try AMD
        vram = get_amd_vram()
        if vram:
            return "AMD", vram
            
        return "Linux/Windows (Unknown GPU)", "Could not detect NVIDIA or AMD GPU."

    return "Unknown OS", "VRAM detection not supported on this OS."


# --- Ollama Management Functions ---
def check_ollama_installed():
    """Checks if Ollama is installed."""
    print("\nChecking if Ollama is installed...")
    command = "command -v ollama" 
    if sys.platform != "win32":
        command = 'command -v ollama'
    else: 
        command = "where ollama"
    return run_command(command.split()) is not None

def install_ollama():
    """Provides instructions to install Ollama."""
    print("\nOllama is not found.")
    print("Please install it by following the instructions on https://ollama.com")
    
    if sys.platform == "darwin": # macOS
        print("\nOn macOS, you can typically install it by running:")
        print("curl -fsSL https://ollama.com/install.sh | sh")
    elif sys.platform == "linux": # Linux
        print("\nOn Linux, you can typically install it by running:")
        print("curl -fsSL https://ollama.com/install.sh | sh")
    else:
        print("\nPlease download the installer from the Ollama website.")

    sys.exit("Exiting script. Please re-run after installing Ollama.")

def serve_ollama():
    """Starts the Ollama server as a background process."""
    print("\nAttempting to start the Ollama server...")
    try:
        command = "ollama serve"
        if sys.platform == "win32":
            subprocess.Popen(f"start /B {command}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        print("Ollama server started as a background process.")
        time.sleep(5)
    except Exception as e:
        print(f"Failed to start Ollama server: {e}")
        sys.exit(1)

'''
# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Detect hardware and VRAM
    gpu_vendor, vram_info = get_system_vram()
    print(f"  - GPU/Platform: {gpu_vendor}")
    print(f"  - Detected VRAM: {vram_info}")

    # 2. Check and manage Ollama installation
    if not check_ollama_installed():
        install_ollama()
    else:
        print("Ollama is already installed.")
    
    # 3. Start the Ollama server
    serve_ollama()
    
    # Your main application logic would continue here...
    print("\nBackend script can now proceed with its main tasks...")'''
