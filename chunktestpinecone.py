from pinecone import Pinecone as PineconeClient
from dotenv import load_dotenv
import os

load_dotenv()
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("faq-vtc")

results = index.query(
    vector=[0.0]*1536,
    top_k=100,
    namespace="vtc-taxi",
    include_metadata=True
)

for match in results["matches"]:
    text = match["metadata"].get("text", "")
    if "octobre" in text.lower() and "convocation" in text.lower():
        print("✅ Chunk trouvé :")
        print(text)
