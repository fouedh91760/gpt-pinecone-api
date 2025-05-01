import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone
from langchain.chains import RetrievalQA

# CONFIGURATION
OPENAI_API_KEY = "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"
INDEX_NAME = "faq-vtc"
NAMESPACE = "default"

# EMBEDDING MODEL
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# VECTOR STORE
vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)

# LLM
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4", temperature=0)

# RETRIEVAL QA CHAIN
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    chain_type="stuff"
)

# INTERFACE EN BOUCLE
while True:
    query = input("❓ Pose ta question (ou 'exit') : ")
    if query.lower() == "exit":
        break
    result = qa_chain.invoke({"query": query})
    print(f"\n💬 Réponse : {result['result']}\n")
