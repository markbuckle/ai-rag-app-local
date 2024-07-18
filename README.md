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

## Split the documents 

Most PDFs are too big to use on their own. We need to split them into manageable chunks using Lanchain's recursive text splitter. 

## Create an embedding for each chunk

Create a function that returns an embedding function as its own file since we will use this function in two separate places: 1) when we create the database itself and 2) when we query the database.

I used AWS Bedrock for my embedding integrations but you can use any of [these embedding models](https://python.langchain.com/v0.1/docs/integrations/text_embedding/).

## update your database

Update your database using [Chroma](https://www.trychroma.com/)

## run your RAG locally

run your files in a terminal: 

```python
python get_embedding_function.py
```
then
```python
python populate_database.py 
```
then add a prompt to your query file:
```python
python query_data.py "who is Andrew Huberman?"
```

# test your RAG 

Quality of answers will depend on:
:point_right: Source material
:point_right: Text splitting strategy
:point_right: LLM model and prompt

## Tutorial video: 

[Python RAG Tutorial (with Local LLMs): AI For Your PDFs](https://www.youtube.com/watch?v=2TJxpyO3ei4)

