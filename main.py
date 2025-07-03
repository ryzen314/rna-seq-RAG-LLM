import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog # Needed for the initial fileBrowser call's internal filedialog
import asyncio
import threading
import os
import sys

# Import functions from vector.py
from vector import fileBrowser_toplevel, generateAIAnswer, setUserQuestion, doesDatabaseExist, set_vectortk_labels

# --- Global Variables for Main Application ---
global root_main_app
global question_entry_widget # Renamed for clarity to avoid conflict with vector.py's `questionWidget` if it existed
global answer_text_widget
global question_display_label
global generating_response_label # Keep global to update text
global ai_processing_lock # To prevent multiple async calls at once

ai_processing_lock = threading.Lock() # Protects against multiple simultaneous AI calls

# --- Async-Tkinter Integration ---
def run_async_in_thread(func, *args):
    """Runs an async function in a separate thread and handles its completion."""
    def _thread_target():
        asyncio.run(func(*args))

    thread = threading.Thread(target=_thread_target)
    thread.daemon = True # Allow the program to exit even if this thread is running
    thread.start()

def on_async_result(result):
    """Callback function to update the Tkinter GUI after async task completes."""
    answer_text_widget.config(state=tk.NORMAL)
    answer_text_widget.delete(1.0, tk.END)
    answer_text_widget.insert(tk.END, result)
    answer_text_widget.config(state=tk.DISABLED)
    generating_response_label.config(text="") # Clear "Generating response..."
    question_entry_widget.config(state=tk.NORMAL) # Re-enable input
    ai_processing_lock.release() # Release the lock

def on_async_error(error_message):
    """Callback for handling errors from async tasks."""
    answer_text_widget.config(state=tk.NORMAL)
    answer_text_widget.delete(1.0, tk.END)
    answer_text_widget.insert(tk.END, f"Error: {error_message}")
    answer_text_widget.config(state=tk.DISABLED)
    generating_response_label.config(text="") # Clear "Generating response..."
    question_entry_widget.config(state=tk.NORMAL) # Re-enable input
    ai_processing_lock.release() # Release the lock

async def _submit_question_to_ai_async_task():
    """The actual async task to submit the question and get response."""
    question = question_entry_widget.get("1.0", "end-1c").strip() # Use .strip() for text widget
    if not question:
        root_main_app.after(0, on_async_result, "Please enter a question.")
        return

    question_display_label.config(text="The question you asked: " + question)
    question_entry_widget.delete('1.0', 'end')
    question_entry_widget.config(state=tk.DISABLED) # Disable input while processing
    setUserQuestion(question)

    generating_response_label.config(text="Generating response...")
    answer_text_widget.config(state=tk.NORMAL)
    answer_text_widget.delete(1.0, tk.END)
    answer_text_widget.insert(tk.END, "AI is thinking...\n")
    answer_text_widget.config(state=tk.DISABLED)

    try:
        answer = await generateAIAnswer()
        root_main_app.after(0, on_async_result, answer) # Schedule GUI update on main thread
    except Exception as e:
        root_main_app.after(0, on_async_error, str(e)) # Schedule error update on main thread

def on_submit_button_click():
    """Event handler for the submit button."""
    if ai_processing_lock.acquire(blocking=False): # Try to acquire lock non-blocking
        run_async_in_thread(_submit_question_to_ai_async_task)
    else:
        generating_response_label.config(text="AI is already processing a request. Please wait.")


# --- Main Application Setup ---
def setup_main_window():
    global root_main_app, question_entry_widget, answer_text_widget, question_display_label, generating_response_label

    root_main_app = tk.Tk()
    root_main_app.geometry("1000x700") # Adjust size for typical use
    root_main_app.title("Dougan Lab RNA-seq database assistant")

    # Hide the main window initially
    root_main_app.withdraw()

    # Call the directory selection dialog (blocking until dialog closes)
    # fileBrowser_toplevel is a new function name in vector.py for clarity
    fileBrowser_toplevel(root_main_app)

    # Check if database setup was successful
    # The 'doesDatabaseExist' function should be updated in vector.py
    # to accurately reflect success/failure from fileBrowser_toplevel
    if doesDatabaseExist(): # This global state is set in vector.py
        root_main_app.deiconify() # Show the main window
    else:
        # If database setup failed (e.g., user canceled), show error and exit
        tk.messagebox.showerror("Setup Error", "Database setup not completed. Application will exit.")
        root_main_app.destroy()
        sys.exit() # Force exit

    # --- GUI Elements for Main Window (after setup) ---
    submit_label = tk.Label(root_main_app, text="Enter your question:")
    submit_label.pack(pady=10)

    question_entry_widget = tk.Text(root_main_app, height=10, width=100)
    question_entry_widget.pack(pady=10)
    
    # You can bind the Return key too if you like (uncomment if needed)
    # question_entry_widget.bind("<Return>", lambda event: on_submit_button_click())

    submit_button = tk.Button(root_main_app, text='Submit question', command=on_submit_button_click)
    submit_button.pack(pady=10)

    answer_text_widget = scrolledtext.ScrolledText(root_main_app, wrap=tk.WORD, width=500, height=300)
    answer_text_widget.pack(pady=10, fill=tk.BOTH, expand=True)

    question_display_label = tk.Label(root_main_app, text="")
    question_display_label.pack(pady=10)

    generating_response_label = tk.Label(root_main_app, text="") # Start empty
    generating_response_label.pack(pady=5)

    root_main_app.mainloop() # Start the main Tkinter event loop


if __name__ == "__main__":
    # Create an asyncio event loop here for the entire application duration
    # This loop will be used by our threads to run async tasks.
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    asyncio.set_event_loop(loop) # Set it as the default loop for this thread (though we're calling it directly)

    # Call the setup function which manages the window flow
    setup_main_window()