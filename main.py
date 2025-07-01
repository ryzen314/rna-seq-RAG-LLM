import os
import sys 
import tkinter as tk 
from tkinter import filedialog
from tkinter import scrolledtext
import asyncio
import tk_async_execute as tae
import vector
from vector import fileBrowser, generateAIAnswer, doesDatabaseExist

global question; question = ""
global results
def submitQuestionToAI():
    results = tae.async_execute(generateAIAnswer(question))


def main():
    global root
    #if doesDatabaseExist():
    root = tk.Tk()
    root.title("Dougan Lab RNA-seq database assistant")
    submitLabel = tk.Label(root, text = "Enter text and press Enter: ")
    submitLabel.pack(pady = 10) 
    global questionWidget
    questionWidget = tk.Text(root, height = 10, width = 100)
    questionWidget.pack(pady = 10)
    questionWidget.bind("<Return>", submitQuestionToAI)
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




