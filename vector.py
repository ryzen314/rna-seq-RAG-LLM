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
import streamlit as st
import ollama
import pandas as pd
from openpyxl import *
import re

global databaseExist    
dir_path = "."
excel_pattern = f"{dir_path}/*.xls*"
#model = os.environ.get("MODEL", "mxbai-embed-large")

#Find all matching files using glob
excel_files = []
for file in glob(excel_pattern):
    excel_files.append(file)
print(excel_files)
#list excel files found in the terminal
# if excel_files:
#     print(f"Found {len(excel_files)} Excel files:")
#     for file in excel_files:
#         print(file)
# else:
#     print("No Excel files found.")
#ollama.pull("mxbai-embed-large")

def selectDirectory():
    global filepath
    filepath = filedialog.askdirectory()
    filepath = os.path.normpath(filepath)
    dirPathLabel = tk.Label(vectortk, text='Entering this directory...')

    if os.path.exists(f"{filepath}/chrome_langchain_db"):
        dirPathLabel.config(text="Database exists. Moving on...")
        databaseExist = True
def fileBrowser():
    global vectortk
    vectortk = tk.Tk()
    vectortk.title('Please select location of directory')
    filepathLabel = tk.Label(vectortk, text="Do you have a pre-existing database or would you like to build one?")
    preexistingButton = tk.Button(vectortk, text='Pre-existing VectorDB', command=selectDirectory)
    generateDBButton = tk.Button(vectortk, text='Generate vectorDB', command=selectDirectory)



db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=db_location
)

if add_documents:
    df = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for i in range(len(excel_files)): 
        #store the file name
        filename = re.search(r'[^/]+$', excel_files[i])
        filename = filename.group(0)
        #create tempfile
        tmp = pd.ExcelFile(excel_files[i])
        sheetnamesdf = []
        with pd.ExcelFile(excel_files[i]) as xls:
            tmp2 = pd.read_excel(xls, sheet_name=None)
            sheetnamesdf = tmp.sheet_names
        #create two new columns - one with the sheet name and one with the file name
        for j in range(len(sheetnamesdf)):
            tmp2[sheetnamesdf[j]].insert(loc=6, column='Sheet Name', value = sheetnamesdf[j])
            tmp2[sheetnamesdf[j]].insert(loc=7, column='File Name', value = filename)
        df.append(tmp2)
        print(df)
    
    documents = []
    ids = []

    for i in range(len(df)):
        #add all the sheetnames from the ith file 
        sheetnamesdf = df[i].keys()
        # first iterate over every sheetname within each file
        for k in range(len(df[i].keys())):
            #within every excel sheet - iterate over every row
            for j, row in df[i][list(sheetnamesdf)[k]].iterrows():
                document=Document(
                    page_content=row["Unnamed: 0"] + " " + "Base Mean" + str(row["baseMean"]) + " " + "Log2 Fold Change" + str(row["log2FoldChange"]) + ' ' + row['Sheet Name'] + ' ' + row["File Name"],
                    id = str(row["Unnamed: 0"] + ' ' + row['Sheet Name'] + ' ' + row['File Name'])
                )
                ids.append(str(row["Unnamed: 0"] + ' ' + row['Sheet Name'] + ' ' + row['File Name']))
                documents.append(document)
                if( j % 5400 == 0):
                    print(f"Reached batch limit - adding 5400 rows to documents....")
                    print(documents)
                    vector_store.add_documents(documents=documents, ids=ids)
                    documents = []
                    ids = []
            print(f"Finished with sheet {list(sheetnamesdf)[k]}")
        
# create retriever
retriever = vector_store.as_retriever()
databaseExist = True

async def generateAIAnswer(question):
    model = os.environ.get("MODEL", "deepseek-r1:14b")
    ollamamodel = ChatOllama(base_url='http://localhost:11434', model=model)
    print(type(ollamamodel))
    print("Model loaded successfully")
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
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    question_answering_chain = create_stuff_documents_chain(ollamamodel, chat_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answering_chain)

    retriever_prompt = (
        """Given a chat history and the latest user question which might reference context in the chat history,
        formulate a standalone question which can be understood without the chat history.
        Do NOT answer the question, just reformulate it if needed otherwise return it as is."""
    )
    chat_history = []
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(ollamamodel, retriever, contextualize_q_prompt)
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(ollamamodel, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    if question: 
        message = rag_chain.invoke({"input": question, "chat_history": chat_history})
        print(message["answer"])
        chat_history.extend(
            [
                HumanMessage(content = question),
                AIMessage(content = message["answer"]),
            ]
        )
        return chat_history
                 
