# this file create the vector database

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


index_name="medical-bot"

##Creating Embeddings for Each of The Text Chunks & storing
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=INDEX_NAME)
