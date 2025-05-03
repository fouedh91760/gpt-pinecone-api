from dotenv import load_dotenv
import os
import pinecone
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment="gcp-europe-west4")
 as PineconeClient

load_dotenv()
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
index = pinecone.Index("faq-vtc")

NAMESPACE = "vtc-taxi"  # ou "arrete-officiel", "examens-2025", etc.

results = index.query(
    vector=[0.0]*1536,
    top_k=200,
    namespace=NAMESPACE,
    include_metadata=True
)

print(f"🔍 Recherche dans le namespace : {NAMESPACE}")

for match in results["matches"]:
    metadata = match["metadata"]
    text = metadata.get("text", "").lower()
    if "certificat médical" in text or "aptitude à la conduite" in text:
        print("-----")
        print("📄 Fichier source :", metadata.get("source", "inconnu"))
        print("🔎 Contenu chunk :\n", metadata.get("text", ""))
