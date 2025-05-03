import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec

# === CHARGEMENT DES VARIABLES D'ENVIRONNEMENT ===
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "gcp-europe-west4"
INDEX_NAME = "faq-vtc"
DOCS_ROOT = Path("C:/Users/fouad/Documents/gpt-pinecone-clean2/docs/VTC TAXI/Examen vtc taxi")

# === INITIALISATION EMBEDDINGS & PINECONE (nouveau SDK) ===
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Crée l'index si nécessaire
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="gcp", region="europe-west4")
    )

pinecone_index = pc.Index(INDEX_NAME)

# === FONCTION DE PARSING DES BLOCS TAGGUÉS ===
def extract_tagged_blocks(content):
    pattern = (
        r"## contexte: (.*?)\s*"
        r"### sous_section: (.*?)\s*"
        r"\*\*Q\s*:\*\*\s*(.*?)\s*"
        r"\*\*R\s*:\*\*\s*(.*?)\s*"
        r"\*\*public\s*:\*\*\s*(.*?)\s*"
        r"\*\*statut\s*:\*\*\s*(.*?)\s*"
        r"---"
    )
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    blocs = []
    for match in matches:
        bloc = {
            "contexte": match[0].strip(),
            "sous_section": match[1].strip(),
            "question": match[2].strip(),
            "reponse": match[3].strip(),
            "public": match[4].strip(),
            "statut": match[5].strip().lower()
        }
        blocs.append(bloc)
    return blocs

# === INDEXATION DANS PINECONE ===
log_entries = {}

for file_path in DOCS_ROOT.glob("*.md"):
    section_name = file_path.stem.lower()

    if file_path.stat().st_size == 0:
        print(f"⏭️ Fichier vide ignoré : {file_path.name}")
        log_entries.setdefault(section_name, []).append((file_path.name, "Ignoré (vide)"))
        continue

    print(f"\n📂 Traitement du fichier : {file_path.name} (namespace = {section_name})")

    try:
        pinecone_index.delete(delete_all=True, namespace=section_name)
        print(f"🧹 Namespace '{section_name}' nettoyé")
    except Exception as e:
        print(f"⚠️ Erreur suppression namespace '{section_name}': {e}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = extract_tagged_blocks(content)
    valid_blocks = [b for b in blocks if b["statut"] == "validé"]

    if not valid_blocks:
        print(f"⚠️ Aucun bloc `validé` trouvé dans : {file_path.name}")
        log_entries.setdefault(section_name, []).append((file_path.name, "0 bloc validé"))
        continue

    texts = [f"Q: {b['question']}\nR: {b['reponse']}" for b in valid_blocks]
    metadatas = [
        {
            "question": b["question"],
            "reponse": b["reponse"],
            "contexte": b["contexte"],
            "sous_section": b["sous_section"],
            "public": b["public"],
            "section": section_name,
            "source": str(file_path)
        }
        for b in valid_blocks
    ]

    try:
        Pinecone.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            index_name=INDEX_NAME,
            namespace=section_name
        )
        print(f"✅ {len(valid_blocks)} blocs indexés dans '{section_name}' depuis {file_path.name}")
        log_entries.setdefault(section_name, []).append((file_path.name, f"{len(valid_blocks)} blocs indexés"))
    except Exception as e:
        print(f"❌ Erreur indexation {file_path.name} : {e}")
        log_entries.setdefault(section_name, []).append((file_path.name, f"Erreur : {e}"))

# === LOG FINAL ===
with open("index_log.txt", "w", encoding="utf-8") as log:
    for ns, files in log_entries.items():
        log.write(f"📂 Namespace : {ns}\n")
        for fname, status in files:
            log.write(f"  - {fname} → {status}\n")
        log.write("\n")

print("\n✅ Indexation terminée. Voir index_log.txt pour le détail.")
