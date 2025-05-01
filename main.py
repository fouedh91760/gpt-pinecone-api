import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

# === CHARGEMENT DES VARIABLES D’ENVIRONNEMENT ===
load_dotenv()

# === CONFIGURATION ===
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENV = os.environ.get("PINECONE_ENV", "gcp-europe-west4")
INDEX_NAME = "faq-vtc"

# === INITIALISATION PINECONE NOUVEAU SDK ===
pc = Pinecone(api_key=PINECONE_API_KEY)


# Vérifie et crée l'index si nécessaire (à faire une seule fois)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region="europe-west4")
    )

# === EMBEDDINGS & LLM ===
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)

# === INITIALISATION FASTAPI ===
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="PEGASE API",
        version="1.0.0",
        description="API pour répondre aux questions liées aux formations PEGASE",
        routes=app.routes,
    )
    openapi_schema["servers"] = [{"url": "https://gpt-pinecone-api.onrender.com"}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/")
def read_root():
    return {"message": "API is running and ready."}

class SearchRequest(BaseModel):
    question: str
    namespace: str = "default"

@app.post("/search_vtc")
def search_vtc(request: SearchRequest):
    print(f"📥 Requête reçue sur /search_vtc : question='{request.question}', namespace='{request.namespace}'")
    try:
        print("🔍 Initialisation du vecteur store Pinecone...")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=request.namespace,
            text_key="text"
        )

        print("🧠 Création du retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

        print("🔗 Construction de la chaîne QA avec LLM...")
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=False
        )

        print("❓ Envoi de la question au modèle...")
        result = qa_chain.invoke({"query": request.question})

        answer = result.get("result", "Aucune réponse.")
        print("✅ Réponse générée :", answer)

        return {
            "question": request.question,
            "namespace": request.namespace,
            "answer": answer
        }

    except Exception as e:
        print("❌ Erreur dans /search_vtc :", str(e))
        raise HTTPException(status_code=500, detail=f"Erreur : {str(e)}")
