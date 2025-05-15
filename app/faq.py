import pandas as pd
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()


faqs_path = Path(__file__).parent / "resources" / "faq_data.csv"
chroma_client = chromadb.Client()
collection_name = "faqs"
groq_client = Groq()


ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def ingest_faq_data(path):
    if collection_name not in [c.name for c in chroma_client.list_collections()]:
        print(f"Ingesting FAQ data into ChromaDB....")
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef
        )

        df=pd.read_csv(path)
        documents = df['question'].to_list()
        metadata = [{'answer':ans} for ans in df['answer'].to_list()]
        ids = [f"id_{i}" for i in range(len(documents))]

        collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        print(f"FAQ Data successfully ingested into Chroma collection: {collection_name}")
    else:
        print(f"Collection {collection_name} already Exists")


def get_relevent_qa(query):
    collection = chroma_client.get_collection(name=collection_name)
    result = collection.query(
        query_texts=[query],
        n_results=2
    )
    return result

def faq_chain(query):
    result = get_relevent_qa(query)
    context = ''.join([r.get('answer') for r in result['metadatas'][0]])

    answer = generate_answer(query,context)
    return answer

def generate_answer(query,context):
    prompt =f''' Given question and contex below, generate the answer based on the contex only. If you don't find the answer inside the contex then say "I don't know". Do not make things up.

    Question:{query}
    context:{context} '''

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        model=os.environ['GROQ_MODEL'],
    )

    return chat_completion.choices[0].message.content



if __name__=="__main__":
    ingest_faq_data(faqs_path)
    query="Do you take cash as a payment option?"
    # result = get_relevent_qa(query)
    answer = faq_chain(query)
    print(answer)

