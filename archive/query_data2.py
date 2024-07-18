import argparse
from botocore.exceptions import ClientError
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
import boto3
import json

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def get_language_model_response(prompt):
    session = boto3.Session(profile_name="MBAdmin")
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    try:
        response = client.invoke_model(
            modelId="amazon.titan-text-premier-v1:0",
            body=json.dumps(
                {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": 512,
                        "temperature": 0.5,
                    },
                },
            ),
            contentType="application/json",
        )
        response_body = json.loads(response["body"].read())
        generated_text = response_body.get("generated_text")
        if generated_text is None:
            raise ValueError("No generated text found in the response.")
        return generated_text
    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke language model. Reason: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = get_language_model_response(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
