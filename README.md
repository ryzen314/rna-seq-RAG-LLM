# rna-seq-RAG-LLM
This application is meant to help you parse through RNA sequencing data that has been analyzed previously and exported to an excel file. Currently, i have bulk RNA seq data supported, but single cell support will be coming soon.

For bulk RNA it grabs your gene names, baseMean, Log2FoldChange, sheetname and file name. I will add an example image to give you an idea of the format for generating the database. 

<img width="518" alt="image" src="https://github.com/user-attachments/assets/ced2efa5-a585-45c3-830b-d44c447760bb" />
please request the AI service to provide you with the filename if you want to query it. 

When selecting a pre-existing database, please select the parent directory that has your chrome_langchain_db folder database.

The first window will tell you your GPU and the amount of VRAM available as this is important for knowing how large of a model you can run. 

If mxbai-embed-large from ollama isn't downloaded, it will auto download it for you and then you can select a pre-existing LLM or download from the list provided. 


