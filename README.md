# AI RAG Local LLM & PDF Tutorial

This tutorial uses similar concepts to [ai-rag-app](https://github.com/markbuckle/ai-rag-app.git) but is a more advanced version. In addition to the first RAG app, this one runs locally using open-source LLMs from AWS, allows you to update your database with new entries, and provides the ability to test/evaluate our AI generated responses.

## Steps

### Install dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 
    - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the enviroment variable path.
      
    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```python
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additonal help if needed. 


2. Now run this command to install dependenies in the `requirements.txt` file. 

```python
pip install -r requirements.txt
```

## Load the data

Gather PDFs that you would like to use as your source material and place them in your data folder.

If you want to load other types of documents, go to [Langchain Document Loaders](https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/) and select a document loader of your choice. There are also 3rd party document loaders [here](https://python.langchain.com/v0.1/docs/integrations/document_loaders/).

## Query the database

Query the Chroma DB.

```python
python query_data.py "How does Alice meet the Mad Hatter?"
```

> You'll also need to set up an OpenAI account (and set the OpenAI key in your environment variable) for this to work.

Tutorial video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami).

