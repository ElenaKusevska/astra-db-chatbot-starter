from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

import os
from dotenv import load_dotenv
from astrapy.db import AstraDBCollection

load_dotenv()

# Grab the Astra token and api endpoint from the environment
token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
keyspace = os.getenv("ASTRA_DB_KEYSPACE")
openai_api_key = os.getenv("OPENAI_API_KEY")
collection_name = os.getenv("ASTRA_DB_COLLECTION_NAME")
dimension = os.getenv("VECTOR_DIMENSION")
model = os.getenv("VECTOR_MODEL")

# langchain openai interface
llm = OpenAI(openai_api_key=openai_api_key)

if not model:
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
else:
    embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key, model=model)

def get_similar_docs(query, number):
    if not keyspace:
        collection = AstraDBCollection(collection_name=collection_name, token=token,
                                       api_endpoint=api_endpoint)
    else:
        collection = AstraDBCollection(collection_name=collection_name, token=token,
                                       api_endpoint=api_endpoint, namespace=keyspace)
    embedding = list(embedding_model.embed_query(query))
    relevant_docs = collection.vector_find(embedding, limit=number)

    docs_contents = [row['answer'] for row in relevant_docs]
    docs_urls = [row['document_id'] for row in relevant_docs]
    return docs_contents, docs_urls


def build_full_prompt(query):
    relevant_docs, urls = get_similar_docs(query, 3)
    docs_single_string = "\n".join(relevant_docs)
    url = urls[0] # set(urls)
    print(url)

    # prompt that is sent to openai using the response from the vector database and the users original query
    prompt = f''' Answer the question posed in the user query section using the provided context. If you don't know the answer, just say that you don't know, don't try to make up an answer. Also remark on whether the provided context was useful in generating the answer and why.
    user_query_boilerplate = USER QUERY: {query}
    document_context_boilerplate = CONTEXT: {docs_single_string}
    final_answer_boilerplate = Final Answer: '''
    return prompt, url


def send_to_openai(full_prompt):
    return llm.invoke(full_prompt)

