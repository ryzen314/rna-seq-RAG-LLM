
import tkinter as tk
import os
import manage_ollama  # Assuming this file exists and has get_system_vram()
import subprocess
import sys

# --- Helper Functions ---

def find_current_models_downloaded():
    """
    Runs 'ollama list' and returns a list of the names of installed models.
    Returns an empty list if the command fails or no models are found.
    """
    try:
        # Corrected: Arguments should be inside the run() parentheses
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            check=True  # Raises an error if the command fails
        )
        output = result.stdout
        model_names = []
        
        # Corrected: Typo `out` changed to `output`
        lines = output.strip().splitlines()[1:] # Skip header
        
        for line in lines:
            model_name = line.split()[0]
            model_names.append(model_name)
            
        return model_names
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Could not run 'ollama list'. Is Ollama installed?")
        return []

# This will hold the final selected model name for the whole application
final_model_selection = None

def get_selected_model():
    """Allows other parts of the app to get the final chosen model."""
    return final_model_selection

# --- Main Tkinter Window Function ---

def gpuollama_toplevel(parent_root):
    """Creates and manages the GPU and Ollama model selection window."""
    
    # Use local variables instead of globals for better encapsulation
    gpuollamatk = tk.Toplevel(parent_root)
    gpuollamatk.geometry("500x450")
    gpuollamatk.title("GPU and Ollama Setup")
    gpuollamatk.grab_set()

    # --- State Management Variables ---
    # A single variable to hold the final choice
    final_model_var = tk.StringVar()
    
    pre_existing_model_var = tk.StringVar()
    downloadable_model_var = tk.StringVar()

    # --- UI Elements ---
    gpu_info_label = tk.Label(gpuollamatk, text='Detecting GPU information...')
    gpu_info_label.pack(pady=10)

    # --- Logic for Callbacks ---
    def update_final_selection_and_label(selected_model):
        """Updates the final selection variable and the display label."""
        global final_model_selection
        final_model_selection = selected_model
        model_label.config(text=f"Model selected: {selected_model}")

    def on_pre_existing_model_change(*args):
        """Callback for when a pre-existing model is selected."""
        selected = pre_existing_model_var.get()
        update_final_selection_and_label(selected)

    def on_downloadable_model_change(*args):
        """Callback for when a downloadable model is selected."""
        selected = downloadable_model_var.get()
        update_final_selection_and_label(selected)

    def enable_preexisting_models():
        pre_existing_models_dd.config(state='normal')
        models_dropdown.config(state='disabled')
        # Clear the other dropdown's selection when switching
        downloadable_model_var.set('') 
        update_final_selection_and_label(pre_existing_model_var.get())

    def enable_downloadable_models():
        pre_existing_models_dd.config(state='disabled')
        models_dropdown.config(state='normal')
        # Clear the other dropdown's selection when switching
        pre_existing_model_var.set('')
        update_final_selection_and_label(downloadable_model_var.get())

    # --- Hardware Detection ---
    gpu, vram = manage_ollama.get_system_vram()
    if gpu == "Unknown OS":
        # Corrected: .config() needs keyword arguments like text=
        gpu_info_label.config(text="Sorry, unknown OS detected. This app is not supported.")
        # Optionally, add a button to close the app
    else:
        gpu_info_label.config(text=f"GPU: {gpu} | VRAM: {vram}")

    # --- Pre-existing Models Section ---
    pre_existing_models_label = tk.Label(gpuollamatk, text='')
    pre_existing_models_label.pack(pady=5)
    
    already_downloaded_models = find_current_models_downloaded()
    if not already_downloaded_models:
        pre_existing_models_label.config(text="No local Ollama models found.")
        # Disable the button if there are no models
        pre_existing_models_button = tk.Button(gpuollamatk, text='Local Models', state='disabled')
    else:
        pre_existing_models_label.config(text="Select from your local models:")
        pre_existing_models_button = tk.Button(gpuollamatk, text='Local Models', command=enable_preexisting_models)
        pre_existing_model_var.set(already_downloaded_models[0]) # Set a default

    pre_existing_models_button.pack(pady=5)
    
    # Handle case where there are no downloaded models for the OptionMenu
    if not already_downloaded_models:
        # Pass a dummy value if the list is empty to prevent an error
        pre_existing_models_dd = tk.OptionMenu(gpuollamatk, pre_existing_model_var, "No models found")
    else:
        pre_existing_models_dd = tk.OptionMenu(gpuollamatk, pre_existing_model_var, *already_downloaded_models)
    
    pre_existing_models_dd.config(state='disabled')
    pre_existing_models_dd.pack(pady=5)
    # Corrected: It's .trace_add, not .add_trace
    pre_existing_model_var.trace_add("write", on_pre_existing_model_change)

    # --- Downloadable Models Section ---
    downloadable_models_button = tk.Button(gpuollamatk, text="Download a New Model", command=enable_downloadable_models)
    downloadable_models_button.pack(pady=5)

    # UPDATED: Re-added the full list of models
    model_options = [
        'deepseek-r1:1.5b',
        'deepseek-r1:7b',
        'deepseek-r1:8b',
        'deepseek-r1:14b',
        'deepseek-r1:32b',
        'deepseek-r1:72b',
        'mistral',
        'mistral-nemo'
    ]
    downloadable_model_var.set(model_options[0]) # Set a default

    models_dropdown = tk.OptionMenu(gpuollamatk, downloadable_model_var, *model_options)
    models_dropdown.config(state='disabled')
    models_dropdown.pack(pady=5)
    downloadable_model_var.trace_add("write", on_downloadable_model_change)

    # --- Final Selection Display ---
    model_label = tk.Label(gpuollamatk, text='')
    model_label.pack(pady=10)

    # --- Done Button ---
    done_button = tk.Button(gpuollamatk, text="Confirm and Continue", command=gpuollamatk.destroy)
    done_button.pack(pady=10)

    # Make the parent window wait until this one is closed
    parent_root.wait_window(gpuollamatk)
