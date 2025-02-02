from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
    logger.info("Pinecone index initialized")
    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
   

# Function to generate embeddings and store in Pinecone
def store_embeddings(chunks, vectorstore):
   try:
      vectorstore.add_documents(chunks)
      return "Embeddings stored successfully"
   except Exception as e:
      logger.error(f"Error storing embeddings: {e}")
      return "Error storing embeddings"
   


def main():
    # Initialize Pinecone index
    vectorstore = initialize_pinecone_index()

    # Example usage
    chunks = ["This is a test chunk", "Another test chunk"]

    docs = [Document(page_content=chunk, metadata={"source": "test"}) for chunk in chunks]
    result = store_embeddings(docs, vectorstore)
    logger.info(result)

if __name__ == "__main__":
    main()