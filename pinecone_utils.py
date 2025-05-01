
import os
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ["PINECONE_ENV"]
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
NAMESPACE = os.environ["PINECONE_NAMESPACE"]

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
