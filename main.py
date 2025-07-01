import os
import sys 
import tkinter as tk 
from tkinter import filedialog
from tkinter import scrolledtext
import tk_async_execute as tae
import vector
from vector import fileBrowser, generateAIAnswer, getUserQuestion
import asyncio



global results
def submitQuestionToAI(event=None):
    question = questionWidget.get("1.0", "end-1c")
    questionLabel.config(text="The question you asked:   " + question)
    event.widget.delete('1.0','end')
    getUserQuestion(question)
    tae.async_execute(generateAIAnswer())


def main():
    global root
    root = tk.Tk()
    root.title("Dougan Lab RNA-seq database assistant")
    submitLabel = tk.Label(root, text = "Enter text and press Enter: ")
    submitLabel.pack(pady = 10) 
    global questionWidget
    questionWidget = tk.Text(root, height = 10, width = 100)
    questionWidget.pack(pady = 10)
    questionWidget.bind("<Return>", submitQuestionToAI)
    global questionLabel; questionLabel = tk.Label(root)
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




