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
    os.makedirs(db_location, exist_ok=True) # Ensure the directory exists

    global embeddings; embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    global vector_store; vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=db_location
    )
    
    excel_files = grabExcelFiles(filepath)

    df_list = []
    for i, excel_file_path in enumerate(excel_files):
        filename = os.path.basename(excel_file_path)
        set_vectortk_labels(dirPathLabel, f"Processing {filename} ({i+1}/{len(excel_files)})...", append=True)
        try:
            with pd.ExcelFile(excel_file_path) as xls:
                file_sheets_data = {}
                for sheet_name in xls.sheet_names:
                    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
                    df_sheet['Sheet Name'] = sheet_name
                    df_sheet['File Name'] = filename
                    file_sheets_data[sheet_name] = df_sheet
                df_list.append(file_sheets_data)
        except Exception as e:
            set_vectortk_labels(dirPathLabel, f"Error reading {filename}: {e}", append=True)
            continue # Skip to next file if there's an error

    documents = []
    ids = []
    
    if df_list: # Only proceed if there's data to process
        set_vectortk_labels(dirPathLabel, "Adding documents to database (this may take a while)...", append=True)

        for file_data in df_list:
            for sheet_name, df_sheet in file_data.items():
                for j, row in df_sheet.iterrows():
                    try:
                        # Ensure required columns exist before accessing
                        if not all(col in row for col in ["Unnamed: 0", "baseMean", "log2FoldChange"]):
                            print(f"Skipping row {j} in sheet {sheet_name} of {row.get('File Name', 'unknown')} due to missing required columns.")
                            continue

                        doc_content = (
                            f"{row['Unnamed: 0']} "
                            f"Base Mean {row['baseMean']} "
                            f"Log2 Fold Change {row['log2FoldChange']} "
                            f"{row['Sheet Name']} "
                            f"{row['File Name']}"
                        )
                        doc_id = f"{row['Unnamed: 0']} {row['Sheet Name']} {row['File Name']}"

                        document = Document(page_content=doc_content, id=doc_id)
                        ids.append(doc_id)
                        documents.append(document)

                        if len(documents) >= 5400: # Batch adding for performance
                            print(f"Adding {len(documents)} rows to ChromaDB...")
                            vector_store.add_documents(documents=documents, ids=ids)
                            documents = []
                            ids = []
                            set_vectortk_labels(dirPathLabel, f"Added batch to DB. Current docs in batch: {len(documents)}", append=True)
                            
                    except Exception as e:
                        set_vectortk_labels(dirPathLabel, f"Error processing row {j} in sheet {sheet_name} of {row.get('File Name', 'unknown')}: {e}", append=True)
                        continue # Continue to next row even if one fails

        # Add any remaining documents after the loop
        if documents:
            print(f"Adding remaining {len(documents)} rows to ChromaDB...")
            vector_store.add_documents(documents=documents, ids=ids)
        set_vectortk_labels(dirPathLabel, "All documents processed and added to database.", append=True)
    else:
        set_vectortk_labels(dirPathLabel, "No data processed for the database.", append=True)
    
    # After generating the vector db - load retriever
    global retriever; retriever = loadVectorDB(filepath)
    set_vectortk_labels(dirPathLabel, "Database generation complete. Retriever loaded.", append=True)


def setUserQuestion(quest):
    global question; question = quest

async def generateAIAnswer():
    model = os.environ.get("MODEL", "deepseek-r1:14b")
    # Add a check for retriever's existence before using it
    if 'retriever' not in globals() or retriever is None:
        return "Error: Database retriever not initialized. Please set up the database first."
        
    try:
        ollamamodel = ChatOllama(base_url='http://localhost:11434', model=model)
        print("Ollama Model loaded successfully (from within bundled app)")
    except Exception as e:
        print(f"Error loading Ollama model: {e}")
        return f"Error connecting to Ollama. Please ensure Ollama is running at http://localhost:11434 with model '{model}'. Error: {e}"
        
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
    
    global chat_history
    
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        ollamamodel, retriever, chat_prompt
    )
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    Youtube_chain = create_stuff_documents_chain(ollamamodel, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)

    if question:
        try:
            response = rag_chain.invoke(
                {"input": question, "chat_history": chat_history},
                config={"configurable": {"session_id": "rna_seq_session"}}
            )
            answer_text = response["answer"]
            chat_history.extend([
                HumanMessage(content=question),
                AIMessage(content=answer_text),
            ])
            return answer_text
        except Exception as e:
            print(f"Error during AI answer generation: {e}")
            return f"An error occurred while generating the AI answer: {e}"
    return "No question was provided."