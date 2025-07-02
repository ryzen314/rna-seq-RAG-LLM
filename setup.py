from cx_Freeze import setup, Executable
import os
# Define the application name and version
app_name = "RNA RAG LLM"
app_version = "0.1"
app_description = "Local LLM queries the rna seq database the user provides"

# Define the executable
executables = [Executable("main.py", base="Console")]
tk_async_execute_path = '/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages'

# Setup configuration
setup(
    name=app_name,
    version=app_version,
    description=app_description,
    options={
        "build_exe": {
            "packages": ["tkinter", "numpy", "pandas", "langchain", "langchain_community", "openpyxl", "langchain_ollama", 
                         "langchain_chroma", "ollama", "asyncio", "threading"], # Include any additional packages your app uses
        },
        "bdist_mac": {
            "bundle_name": app_name,
        },
    },
    executables=executables,
)
