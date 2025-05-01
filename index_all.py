import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# === CHARGEMENT DES VARIABLES D'ENVIRONNEMENT ===
load_dotenv()

# === CONFIGURATION ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "gcp-europe-west4"
INDEX_NAME = "faq-vtc"
DOCS_ROOT = Path("docs")

# === INIT EMBEDDINGS & PINECONE ===
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
pc = PineconeClient(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region=PINECONE_ENV)
    )

pinecone_index = pc.Index(INDEX_NAME)
processed_namespaces = set()
log_entries = {}

# === PARSEUR Q/R ===
def extract_question_answer_blocks(text):
    pattern = r"###\s+(.*?)\n\*\*Réponse\s*:\*\*\s*(.+?)(?=\n###|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)
    return [{"question": q.strip(), "reponse": r.strip()} for q, r in matches]

# === INDEXATION ===
for file_path in DOCS_ROOT.rglob("*.md"):
    if file_path.stat().st_size == 0:
        print(f"⏭️ Fichier vide ignoré : {file_path.name}")
        namespace = file_path.relative_to(DOCS_ROOT).parts[0].lower().replace(" ", "-")
        log_entries.setdefault(namespace, []).append((file_path.name, "Ignoré (vide)"))
        continue

    # Namespace dynamique basé sur le dossier
    raw_namespace = file_path.relative_to(DOCS_ROOT).parts[0]
    namespace = re.sub(r'[^a-zA-Z0-9_-]', '', raw_namespace.lower().replace(" ", "-"))

    print(f"\n📂 Traitement du namespace : '{namespace}' (depuis dossier : {raw_namespace})")

    if namespace not in processed_namespaces:
        try:
            pinecone_index.delete(delete_all=True, namespace=namespace)
            print(f"🧹 Namespace '{namespace}' supprimé avant réindexation")
        except Exception as e:
            print(f"⚠️ Erreur suppression namespace '{namespace}': {e}")
        processed_namespaces.add(namespace)

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = extract_question_answer_blocks(content)
    if not blocks:
        print(f"⚠️ Aucun bloc Q/R trouvé dans : {file_path.name}")
        log_entries.setdefault(namespace, []).append((file_path.name, "Aucun bloc Q/R trouvé"))
        continue

    texts = [f"### {b['question']}\nRéponse : {b['reponse']}" for b in blocks]
    metadatas = [{"question": b["question"], "reponse": b["reponse"], "source": str(file_path.relative_to(DOCS_ROOT))} for b in blocks]

    try:
        Pinecone.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            index_name=INDEX_NAME,
            namespace=namespace
        )
        print(f"✅ {len(texts)} Q/R indexées dans '{namespace}' depuis {file_path.name}")
        log_entries.setdefault(namespace, []).append((file_path.name, f"{len(texts)} Q/R"))
    except Exception as e:
        print(f"❌ Erreur indexation {file_path.name} : {e}")
        log_entries.setdefault(namespace, []).append((file_path.name, f"Erreur : {e}"))

# === ENREGISTREMENT DU RÉCAP ===
with open("index_log.txt", "w", encoding="utf-8") as log:
    for ns, files in log_entries.items():
        log.write(f"📂 Namespace : {ns}\n")
        for fname, status in files:
            log.write(f"  - {fname} → {status}\n")
        log.write("\n")

print("\n✅ Indexation Q/R terminée. Voir index_log.txt pour le détail.")
