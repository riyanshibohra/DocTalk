from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from dotenv import load_dotenv
import os
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create or get index
def initialize_pinecone_index(index_name="doctalk"):
    # Create index if it doesn't exist
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    # Initialize the Pinecone vector store
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        text_key="text"
    )
    print("Pinecone index initialized")

# Function to generate embeddings and store in Pinecone
def store_embeddings(chunks, vectorstore):
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed(chunk)
        vectorstore.upsert([(f'chunk-{i}', embedding)]) 