import os
from glob import glob
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from openpyxl import load_workbook # Changed from `from openpyxl import *` for better practice
import re
import tkinter as tk
from tkinter import filedialog
import asyncio

# --- Global Flags/Objects for Shared State ---
global databaseExist; databaseExist = False
global generateDB # This is now determined by user choice, no need for separate global
global retriever
global embeddings
global vector_store
global question # Set by main.py
global vectortk # The Toplevel window for directory selection
global dirPathLabel # Label within vectortk to update messages


chat_history = [] # Shared chat history for LangChain


# --- Tkinter Label Helper (can be put in main.py or a util file too) ---
def set_vectortk_labels(labelObj, string, append = False):
    if append:
        labelObj['text'] += "\n" + string # Added newline for better display
    else:
        labelObj['text'] = string
    # Force update the GUI immediately (useful during long ops)
    if vectortk and vectortk.winfo_exists(): # Check if window still exists
        vectortk.update_idletasks()


def doesDatabaseExist():
    """Returns the current state of database existence flag."""
    return databaseExist


# Modified to accept parent_root
def selectDirectory_action(parent_root):
    global databaseExist, generateDB, retriever
    # filedialog.askdirectory returns the path, or an empty string if canceled
    chosen_filepath = filedialog.askdirectory()
    
    if not chosen_filepath: # User cancelled selection
        set_vectortk_labels(dirPathLabel, "Directory selection cancelled. Please try again.")
        # Optionally, close the dialog here if cancellation means exiting setup
        # vectortk.destroy()
        # global databaseExist; databaseExist = False # Ensure flag reflects cancellation
        return # Do not proceed if directory not selected

    # Update dirPathLabel with a loading message immediately
    set_vectortk_labels(dirPathLabel, "Checking directory...")

    chosen_filepath = os.path.normpath(chosen_filepath)
    
    global databaseExist # Declare global usage
    try:
        if os.path.exists(f"{chosen_filepath}/chrome_langchain_db"):
            set_vectortk_labels(dirPathLabel, "Database exists. Loading...")
            generateDB = False
            global retriever; retriever = loadVectorDB(chosen_filepath)
            databaseExist = True # Set global flag
        else:
            set_vectortk_labels(dirPathLabel, "Database not found. Generating...")
            generateDB = True
            generateDatabase(chosen_filepath) # Pass chosen_filepath to generation
            databaseExist = True # Set global flag after generation
        
        # If successfully loaded or generated, destroy this Toplevel window
        if vectortk and vectortk.winfo_exists():
            vectortk.destroy() # Destroy only the Toplevel dialog

    except Exception as e:
        set_vectortk_labels(dirPathLabel, f"An error occurred during database setup: {e}\nPlease try again.", append=True)
        databaseExist = False # Mark as failed

# In vector.py

def append_to_database_action(parent_root):
  
    global retriever, vector_store, databaseExist

    # 1. Ask user for the existing database directory
    set_vectortk_labels(dirPathLabel, "First, please select the directory containing your existing database.")
    db_filepath = filedialog.askdirectory(title="Select Existing Database Directory")
    if not db_filepath or not os.path.exists(os.path.join(db_filepath, "chrome_langchain_db")):
        set_vectortk_labels(dirPathLabel, "Invalid directory or no database found here. Operation cancelled.")
        return

    # 2. Load the existing VectorDB
    set_vectortk_labels(dirPathLabel, "Loading existing database...")
    try:
        # loadVectorDB returns the retriever, but also sets the global vector_store
        retriever = loadVectorDB(db_filepath)
        set_vectortk_labels(dirPathLabel, "Database loaded successfully.", append=True)
    except Exception as e:
        set_vectortk_labels(dirPathLabel, f"Error loading database: {e}", append=True)
        return

    # 3. Ask user for the NEW Excel files to add
    set_vectortk_labels(dirPathLabel, "\nNow, please select the NEW Excel file(s) to append.", append=True)
    new_excel_files = filedialog.askopenfilenames(
        title="Select NEW Excel Files to Append",
        filetypes=[("Excel files", "*.xlsx *.xls")]
    )
    if not new_excel_files:
        set_vectortk_labels(dirPathLabel, "No new files selected. Operation cancelled.", append=True)
        return

    # 4. Use the generator and batching logic to add new documents
    batch_size = 5400
    batch_docs = []
    batch_ids = []
    total_docs_added = 0

    # Loop over the generator to process new files one by one
    for doc, doc_id in process_excel_to_docs(list(new_excel_files), dirPathLabel):
        batch_docs.append(doc)
        batch_ids.append(doc_id)

        # If the batch is full, add it to the database
        if len(batch_docs) >= batch_size:
            set_vectortk_labels(dirPathLabel, f"Appending batch of {len(batch_docs)} documents...", append=True)
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            total_docs_added += len(batch_docs)
            batch_docs, batch_ids = [], [] # Reset the batch

    # Add any leftover documents after the loop
    if batch_docs:
        set_vectortk_labels(dirPathLabel, f"Appending final batch of {len(batch_docs)} documents...", append=True)
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        total_docs_added += len(batch_docs)

    # 5. Finalize the process
    if total_docs_added > 0:
        set_vectortk_labels(dirPathLabel, f"\nAppend complete! Added {total_docs_added} new documents.", append=True)
    else:
        set_vectortk_labels(dirPathLabel, "\nNo new documents were processed or added.", append=True)

    # Mark database as ready and close the dialog
    databaseExist = True
    vectortk.destroy()
# Entry point for the Toplevel dialog, called from main.py

def fileBrowser_toplevel(parent_root):
    global vectortk, dirPathLabel
    vectortk = tk.Toplevel(parent_root) # Create as a Toplevel child of the main app root
    vectortk.geometry("500x400")
    vectortk.title('Please select location of database directory')
    
    # Make this dialog modal and wait for it to close
    # This prevents interaction with the main window until this dialog is handled.
    vectortk.grab_set()
    
    filepathLabel = tk.Label(vectortk, text="Do you have a pre-existing database or would you like to build one?")
    filepathLabel.pack(pady=10)

    # Use lambda to pass parent_root to the action function
    preexistingButton = tk.Button(vectortk, text='Select Pre-existing VectorDB Directory', command=lambda: selectDirectory_action(parent_root))
    preexistingButton.pack(padx=10, pady=5)

    generateDBButton = tk.Button(vectortk, text='Generate VectorDB from Directory', command=lambda: selectDirectory_action(parent_root))
    generateDBButton.pack(padx=10, pady=5)
    appendButton = tk.Button(vectortk, text = 'Append New Excel Files to Exisiting DB', command=lambda: append_to_database_action(parent_root))
    appendButton.pack(padx=10, pady=5)

    dirPathLabel = tk.Label(vectortk, text="") # Initialize this label
    dirPathLabel.pack(pady=10)

    # This will block the parent_root's execution until vectortk is destroyed
    parent_root.wait_window(vectortk)
    # Execution in main.py will resume here after vectortk closes.


def loadVectorDB(filepath):
    db_location = f"{filepath}/chrome_langchain_db"
    print(f'Loading vector db from: {db_location}...')
    global embeddings; embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    print('Loading embeddings...')
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=db_location
    )
    # create retriever
    retriever = vector_store.as_retriever()
    print('Retriever loaded.')
    return retriever


def process_excel_to_docs(excel_files, vectortk_label):
    """
    Generator function that reads Excel files and yields
    a (Document, doc_id) tuple for each row.
    """
    if vectortk_label:
        set_vectortk_labels(vectortk_label, f"Starting to process {len(excel_files)} new file(s)...")

    for file_path in excel_files:
        filename = os.path.basename(file_path)
        if vectortk_label:
            set_vectortk_labels(vectortk_label, f"Reading: {filename}", append=True)

        try:
            with pd.ExcelFile(file_path) as xls:
                for sheet_name in xls.sheet_names:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                    # We need to add these columns for the generator to use them
                    df_sheet['Sheet Name'] = sheet_name
                    df_sheet['File Name'] = filename
                    for _, row in df_sheet.iterrows():
                        doc_content = (
                            f"{row.get('Unnamed: 0', '')} "
                            f"Base Mean {row.get('baseMean', '')} "
                            f"Log2 Fold Change {row.get('log2FoldChange', '')} "
                            f"{row.get('Sheet Name', '')} "
                            f"{row.get('File Name', '')}"
                        )
                        doc_id = f"{row.get('Unnamed: 0', '')} {row.get('Sheet Name', '')} {row.get('File Name', '')}"
                        
                        # Yield the Document and its ID together
                        yield Document(page_content=doc_content), doc_id
        except Exception as e:
            if vectortk_label:
                set_vectortk_labels(vectortk_label, f"Error processing {filename}: {e}", append=True)
            continue

def grabExcelFiles(filepath):
    excel_pattern = f"{filepath}/*.xls*"
    excel_files = []
    found_any_excel = False
    for file in glob(excel_pattern):
        excel_files.append(file)
        found_any_excel = True

    if found_any_excel:
        set_vectortk_labels(dirPathLabel, f"We found {len(excel_files)} Excel files in the directory:")
        for file in excel_files:
            set_vectortk_labels(dirPathLabel, f"  - {os.path.basename(file)}", append=True) # Show only basename
    else:
        set_vectortk_labels(dirPathLabel, "No Excel files found in the selected directory. Generating empty DB or using existing.", append=True)
        # Handle this case: perhaps return empty list or raise an error if files are strictly required
        # For now, it will proceed with an empty `df_list` which might lead to an empty vector_store.
    return excel_files


def generateDatabase(filepath):
    db_location = f"{filepath}/chrome_langchain_db"
    os.makedirs(db_location, exist_ok=True)

    global embeddings; embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    global vector_store; vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=db_location
    )
    
    excel_files = grabExcelFiles(filepath)
    if not excel_files:
        set_vectortk_labels(dirPathLabel, "No Excel files found to generate database", append=True)
        return


    batch_size = 5400
    batch_docs = []
    batch_ids = []
    total_docs_added = 0

    # CORRECT: Loop over the generator to get each document as it's created.
    # The generator yields a (doc, doc_id) tuple, which we unpack here.
    for doc, doc_id in process_excel_to_docs(excel_files, dirPathLabel):
        batch_docs.append(doc)
        batch_ids.append(doc_id)

        # Check if the batch is full
        if len(batch_docs) >= batch_size:
            print(f"Adding batch of {len(batch_docs)} to ChromaDB...")
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            total_docs_added += len(batch_docs)
            
            # Clear the lists for the next batch
            batch_docs = []
            batch_ids = []

    # Add any leftover documents that didn't make a full batch
    if batch_docs:
        print(f"Adding final batch of {len(batch_docs)} to ChromaDB...")
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        total_docs_added += len(batch_docs)

    set_vectortk_labels(dirPathLabel, f"Database generation complete! Added {total_docs_added} total documents.", append=True)
    
    # After generating the vector db - load retriever
    global retriever; retriever = loadVectorDB(filepath)
    set_vectortk_labels(dirPathLabel, "Retriever is ready.", append=True)
def setUserQuestion(quest):
    global question; question = quest

# In vector.py (within generateAIAnswer)

# In vector.py (within generateAIAnswer)
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter # Need to import itemgetter

# ... (rest of imports and global definitions)

# Async def generateAIAnswer():
# ... (model setup and system_prompt, history_aware_prompt, etc.)

# Define the session store and get_session_history function (as before)
_session_store: dict[str, list[HumanMessage | AIMessage]] = {}
def _get_session_history_for_chain(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = []
    return ChatMessageHistory(messages=_session_store[session_id])

async def generateAIAnswer():
    model = os.environ.get("MODEL", "deepseek-r1:14b")
    if 'retriever' not in globals() or retriever is None:
        return "Error: Database retriever not initialized. Please set up the database first."
        
    try:
        ollamamodel = ChatOllama(base_url='http://localhost:11434', model=model)
        print("Ollama Model loaded successfully (from within bundled app)")
    except Exception as e:
        print(f"Error loading Ollama model: {e}")
        return f"An error occurred while generating the AI answer. Error: {e}"
        
    system_prompt = (
        """You are an assistant for querying RNA sequencing database and
        providing information regarding genes that are differentially regulated.
        Base Mean is the expression level and Log2FoldChange is the fold change relative
        to the control condition. Use the following pieces from the retrieved database to
        answer your question.
        If you don't know the answer, say you don't know.
        \n\n
        {context}
        """
    )

    # 1. Prompt for History-Aware Retriever (reformulates question)
    history_aware_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Given a chat history and the latest user question which might reference context in the chat history,"
             "formulate a standalone question which can be understood without the chat history."
             "Do NOT answer the question, just reformulate it if needed otherwise return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever_chain = create_history_aware_retriever(ollamamodel, retriever, history_aware_prompt)

    # 2. Prompt for Answering Question (uses context)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt), # This has {context}
            MessagesPlaceholder("chat_history"), # This has {chat_history}
            ("human", "{input}"),
        ]
    )
    Youtube_chain = create_stuff_documents_chain(ollamamodel, qa_prompt)

    # 3. Combine them into a RAG chain
    # The create_retrieval_chain passes 'context' from retriever to stuff_documents_chain.
    # This is where the context is generated.
    retrieval_rag_chain = create_retrieval_chain(history_aware_retriever_chain, Youtube_chain)

    # --- NEW CRITICAL STEP: Ensure 'context' is explicitly passed from the retriever's output
    # through the RunnableWithMessageHistory wrapper.
    # The output of `retrieval_rag_chain` is a dict that includes "context" and "answer".
    # We need to map these to the final prompt's expected variables when wrapping.

    # When using `RunnableWithMessageHistory`, the `retrieval_rag_chain` is its `runnable`.
    # The `retrieval_rag_chain` will output a dictionary like `{"input": ..., "chat_history": ..., "context": ..., "answer": ...}`.
    # We need to ensure that the `context` is accessible by the `qa_prompt`.
    # A common way to handle this when composing chains is to use `RunnablePassthrough.assign`
    # or `itemgetter` on the inputs to the `stuff_documents_chain`.

    # However, since `create_retrieval_chain` is designed to pass context correctly to
    # `create_stuff_documents_chain`, the issue is often related to `RunnableWithMessageHistory`
    # not exposing `context` through its `invoke` parameters correctly for nested chains.

    # Let's rebuild the `conversational_rag_chain` using a standard pattern from LangChain docs
    # that correctly exposes `context` and `chat_history`.

    # Define the core RAG chain that takes `input` and `chat_history` (and generates `context`)
    # and produces `answer`.
    # This chain will output `{"answer": ..., "context": ..., "chat_history": ...}`
    # We then use RunnableWithMessageHistory to inject the history.

    final_combined_chain = (
        history_aware_retriever_chain # First, retrieve documents (outputs {"context": docs, "input": ..., "chat_history": ...})
        | Youtube_chain # Then, answer using the retrieved context and other inputs
                               # Youtube_chain expects {context, input, chat_history}
    )

    # Wrap the final chain with RunnableWithMessageHistory for automatic chat history management
    conversational_rag_chain = RunnableWithMessageHistory(
        final_combined_chain, # The chain that takes input & chat_history and uses context
        _get_session_history_for_chain,
        input_messages_key="input", # User's current message
        history_messages_key="chat_history", # Placeholder name in your prompt
        # We don't need output_messages_key if the chain's direct output is not a BaseMessage list
        # And context_messages_key isn't a direct parameter of RunnableWithMessageHistory itself.
    )

    if question:
        try:
            # Invoke the wrapped chain. It manages input, chat_history, and context internally.
            response = await conversational_rag_chain.ainvoke(
                {"input": question}, # Only pass the current input
                config={"configurable": {"session_id": "rna_seq_session"}}
            )
            answer_text = response["answer"]
            # No manual history.extend needed here, as RunnableWithMessageHistory handles it.
            return answer_text
        except Exception as e:
            print(f"Error during AI answer generation: {e}")
            return f"An error occurred while generating the AI answer: {e}"
    return "No question was provided."
