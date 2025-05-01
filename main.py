from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import os
from dotenv import load_dotenv

# === CHARGEMENT DES VARIABLES D’ENVIRONNEMENT ===
load_dotenv()

# === CONFIGURATION ===
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = "gcp-europe-west4"
INDEX_NAME = "faq-vtc"

# === INITIALISATION PINECONE & EMBEDDINGS ===
pc = PineconeClient(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='gcp', region=PINECONE_ENV)
    )

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)

# === INITIALISATION DE L’API FASTAPI ===
app = FastAPI()

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="PEGASE API",
        version="1.0.0",
        description="API pour répondre aux questions liées aux formations PEGASE",
        routes=app.routes,
    )
    openapi_schema["servers"] = [
        {"url": "https://gpt-pinecone-api.onrender.com"}
    ]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# === ROUTE DE TEST DE SANTÉ ===
@app.get("/")
def read_root():
    return {"message": "API is running and ready."}

# === STRUCTURE DE LA REQUÊTE ===
class SearchRequest(BaseModel):
    question: str
    namespace: str = "default"

# === ROUTE DE RECHERCHE PRINCIPALE ===
@app.post("/search_vtc")
def search_vtc(request: SearchRequest):
    try:
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=request.namespace,
            text_key="text"
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )
        result = qa_chain.invoke({"query": request.question})
        answer = result.get("result", "Aucune réponse.")
        return {
            "question": request.question,
            "namespace": request.namespace,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")
