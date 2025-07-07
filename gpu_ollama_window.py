import tkinter as tk
import os
import manage_ollama  # Import your updated backend script
import sys
import threading
import queue # Import the queue module

# This will hold the final selected model name for the whole application
final_model_selection = None

def get_selected_model():
    """Allows other parts of the app to get the final chosen model."""
    return final_model_selection

# --- Main Tkinter Window Function ---

def gpuollama_toplevel(parent_root):
    """Creates and manages the GPU and Ollama model selection window."""
    
    gpuollamatk = tk.Toplevel(parent_root)
    gpuollamatk.geometry("500x500")
    gpuollamatk.title("GPU and Ollama Setup")
    gpuollamatk.grab_set()

    # --- NEW: Queue for thread-safe communication ---
    q = queue.Queue()

    # --- State Management Variables ---
    pre_existing_model_var = tk.StringVar()
    downloadable_model_var = tk.StringVar()

    # --- 1. DEFINE ALL WIDGETS FIRST ---
    gpu_info_label = tk.Label(gpuollamatk, text='Detecting GPU information...')
    gpu_info_label.pack(pady=10)
    
    status_label = tk.Label(gpuollamatk, text="") # For showing download progress
    status_label.pack(pady=5)

    model_label = tk.Label(gpuollamatk, text='')
    model_label.pack(pady=5)
    
    pre_existing_models_label = tk.Label(gpuollamatk, text='')
    pre_existing_models_label.pack(pady=5)
    
    pre_existing_models_button = tk.Button(gpuollamatk, text='Local Models')
    pre_existing_models_button.pack(pady=5)
    
    pre_existing_models_dd = tk.OptionMenu(gpuollamatk, pre_existing_model_var, "")
    pre_existing_models_dd.pack(pady=5)

    downloadable_models_button = tk.Button(gpuollamatk, text="Download a New Model")
    downloadable_models_button.pack(pady=5)

    model_options = [
        'deepseek-r1:1.5b', 'deepseek-r1:7b', 'deepseek-r1:8b', 'deepseek-r1:14b',
        'deepseek-r1:32b', 'deepseek-r1:72b', 'mistral', 'mistral-nemo'
    ]
    models_dropdown = tk.OptionMenu(gpuollamatk, downloadable_model_var, *model_options)
    models_dropdown.pack(pady=5)
    
    confirm_button = tk.Button(gpuollamatk, text="Confirm and Continue")
    confirm_button.pack(pady=20)
    
    # --- 2. DEFINE ALL HELPER FUNCTIONS AND CALLBACKS ---
    
    def _process_queue():
        """
        Checks the queue for messages from the worker thread and updates the GUI.
        This function always runs on the main thread.
        """
        try:
            message = q.get_nowait()
            # Process the message based on its 'status'
            if message['status'] == 'downloading':
                status_label.config(text=f"Downloading {message['model']}... Please wait.")
                # Your new logic: don't let the user proceed if the dependency is downloading
                if message['model'].startswith('mxbai-embed-large'):
                    confirm_button.config(state='disabled')
            elif message['status'] == 'success':
                status_label.config(text=f"Successfully downloaded {message['model']}!")
                # Your new logic: if it's the embedder, re-enable the button. Otherwise, close.
                if message['model'].startswith('mxbai-embed-large'):
                    confirm_button.config(state='normal')
                else:
                    gpuollamatk.after(1000, gpuollamatk.destroy)
            elif message['status'] == 'error':
                status_label.config(text=f"Failed to pull model: {message['error']}")
                confirm_button.config(state='normal') # Re-enable button on failure
        except queue.Empty:
            pass
        finally:
            gpuollamatk.after(100, _process_queue)

    def _pull_model_thread(model_to_pull, q_ref):
        """
        This function runs in a separate thread. It does the long work
        and puts messages into the queue instead of touching the GUI.
        """
        q_ref.put({'status': 'downloading', 'model': model_to_pull})
        success = manage_ollama.pull_model(model_to_pull)
        if success:
            q_ref.put({'status': 'success', 'model': model_to_pull})
        else:
            q_ref.put({'status': 'error', 'error': f"Failed to pull {model_to_pull}"})

    def confirm_and_continue():
        if not final_model_selection:
            status_label.config(text="Please select a model first.")
            return

        if final_model_selection in already_downloaded_models:
            gpuollamatk.destroy()
        else:
            confirm_button.config(state='disabled')
            download_thread = threading.Thread(
                target=_pull_model_thread, 
                args=(final_model_selection, q)
            )
            download_thread.start()
            _process_queue()

    def update_final_selection_and_label(selected_model):
        global final_model_selection
        if selected_model:
            final_model_selection = selected_model
            model_label.config(text=f"Model selected: {selected_model}")
            if selected_model in already_downloaded_models:
                confirm_button.config(text="Confirm and Continue")
            else:
                confirm_button.config(text="Download and Continue")

    def on_pre_existing_model_change(*args):
        update_final_selection_and_label(pre_existing_model_var.get())

    def on_downloadable_model_change(*args):
        update_final_selection_and_label(downloadable_model_var.get())

    def enable_preexisting_models():
        pre_existing_models_dd.config(state='normal')
        models_dropdown.config(state='disabled')
        downloadable_model_var.set('') 
        update_final_selection_and_label(pre_existing_model_var.get())

    def enable_downloadable_models():
        pre_existing_models_dd.config(state='disabled')
        models_dropdown.config(state='normal')
        pre_existing_model_var.set('')
        update_final_selection_and_label(downloadable_model_var.get())
        
    def pull_mxbai_embed():
        """Checks for and pulls the embedding model if it's missing."""
        if 'mxbai-embed-large:latest' not in already_downloaded_models:
            download_thread = threading.Thread(
                target=_pull_model_thread, 
                args=('mxbai-embed-large:latest', q)
            )
            download_thread.start()
            _process_queue()

    # --- 3. CONFIGURE WIDGETS AND RUN INITIAL LOGIC ---
    
    # Configure buttons with their commands now that the functions exist
    pre_existing_models_button.config(command=enable_preexisting_models)
    downloadable_models_button.config(command=enable_downloadable_models)
    confirm_button.config(command=confirm_and_continue)
    
    # Configure dropdowns
    downloadable_model_var.set(model_options[0])
    models_dropdown.config(state='disabled')
    downloadable_model_var.trace_add("write", on_downloadable_model_change)

    # Initial Ollama Check
    if not manage_ollama.check_ollama_installed():
        installOllamaLabel = tk.Label(gpuollamatk, text="Ollama is not installed.\nPlease visit ollama.com to install, then restart this application.", justify=tk.LEFT, wraplength=480)
        installOllamaLabel.pack(pady=20, padx=10)
        return
    else:
        manage_ollama.serve_ollama()

    # Hardware Detection
    gpu, vram = manage_ollama.get_system_vram()
    gpu_info_label.config(text=f"GPU: {gpu} | VRAM: {vram}")

    # Pre-existing Models Section
    already_downloaded_models = manage_ollama.find_current_models_downloaded()
    
    if not already_downloaded_models:
        pre_existing_models_label.config(text="No local Ollama models found.")
        pre_existing_models_button.config(state='disabled')
        pre_existing_models_dd.config(state='disabled')
    else:
        pre_existing_models_label.config(text="Select from your local models:")
        # Update the OptionMenu with the found models
        menu = pre_existing_models_dd['menu']
        menu.delete(0, 'end')
        for model in already_downloaded_models:
            menu.add_command(label=model, command=tk._setit(pre_existing_model_var, model))
        pre_existing_model_var.set(already_downloaded_models[0])
        pre_existing_models_dd.config(state='disabled')
    
    pre_existing_model_var.trace_add("write", on_pre_existing_model_change)
    
    # Initialize the selection label and button text
    update_final_selection_and_label(pre_existing_model_var.get() or downloadable_model_var.get())
    
    # Check for and pull the embedding model if needed
    pull_mxbai_embed()

    parent_root.wait_window(gpuollamatk)
