import os
import sys 
import tkinter as tk 
from tkinter import filedialog
from tkinter import scrolledtext
#import tk_async_execute as tae
from vector import fileBrowser, generateAIAnswer, setUserQuestion
import asyncio
import threading



global results

def _asyncio_thread(async_loop):
    async_loop.run_until_complete(submitQuestionToAI())

def do_async_tasks(async_loop):
    print(type(async_loop))
    threading.Thread(target=_asyncio_thread, args=(async_loop,)).start()
    


async def submitQuestionToAI(event=None):
    question = questionWidget.get("1.0", "end-1c")
    questionLabel.config(text="The question you asked:   " + question)
    questionWidget.delete('1.0', 'end')
    setUserQuestion(question)
    asyncAITask = await asyncio.create_task(generateAIAnswer())
    #tae.async_execute(generateAIAnswer())


def main(async_loop):
    print(type(async_loop))
    global root
    root = tk.Tk()
    root.geometry("1440x900")
    root.title("Dougan Lab RNA-seq database assistant")
    submitLabel = tk.Label(root, text = "Enter text and press Enter: ")
    submitLabel.pack(pady = 10) 
    global questionWidget
    questionWidget = tk.Text(root, height = 10, width = 100)
    questionWidget.pack(pady = 10)
    #questionWidget.bind("<Return>", lambda async_loop: do_async_tasks(async_loop))
    submitQuestion = tk.Button(root, text='Submit question', command=lambda:do_async_tasks(async_loop))
    submitQuestion.pack(pady=10)
    global questionLabel; questionLabel = tk.Label(root)
    questionLabel.pack(pady = 10)
    global generatingResponseLabel
    generatingResponseLabel = tk.Label(root, text = "Generating response...")
    global resultOutput
    #tae.start()
    root.mainloop()
    #tae.stop()
 

if __name__  == "__main__":
    fileBrowser()
    async_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(async_loop)
    print(type(async_loop))
    main(async_loop)




