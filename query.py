from langchain_pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# CONFIG
import os
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
INDEX_NAME = "faq-vtc"
NAMESPACE = "default"  # adapte si tu changes

# EMBEDDINGS
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

# VECTORSTORE
vectorstore = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)

# RETRIEVER + LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")  # ou gpt-3.5-turbo

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# DEMO QUESTION
while True:
    query = input("❓ Pose ta question (ou tape 'exit') : ")
    if query.lower() == "exit":
        break
    answer = qa.run(query)
    print(f"\n💬 Réponse : {answer}\n")
