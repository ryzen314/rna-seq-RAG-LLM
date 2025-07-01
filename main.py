import os
import sys 
import tkinter as tk 
from tkinter import filedialog
from tkinter import scrolledtext
import asyncio
import tk_async_execute as tae
import vector
from vector import generateAIAnswer, databaseExist

global question; question = ""
global results
def submitQuestionToAI():
    results = tae.async_execute(generateAIAnswer)


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
def main():
    global root
    root = tk.Tk()
    root.title("Dougan Lab RNA-seq database assistant")
    submitLabel = tk.Label(root, text = "Enter text and press Enter: ")
    submitLabel.pack(pady = 10) 
    global questionWidget
    questionWidget = tk.Text(root, height = 10, width = 100)
    questionWidget.pack(pady = 10)
    questionWidget.bind("<Return", submitQuestionToAI)
    global questionLabel; questionLabel = tk.Label(root, text = question)
    questionLabel.pack(pady = 10)
    global generatingResponseLabel
    generatingResponseLabel = tk.Label(root, text = "Generating response...")
    global resultOutput
    tae.start()
    root.mainloop()
    tae.stop()

if __name__  == "__main__":
    fileBrowser()
    main()




