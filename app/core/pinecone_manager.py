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

class PineconeManager:
    def __init__(self, index_name="doctalk"):
        self.index_name = index_name
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.vectorstore = self.initialize_pinecone_index()

    def initialize_pinecone_index(self):
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        logger.info("Pinecone index initialized")
        return PineconeVectorStore(
            index_name=self.index_name,
            embedding=embeddings
        )

    def delete_all_vectors(self):
        """Delete all vectors from the index"""
        try:
            index = self.pc.Index(self.index_name)
            index.delete(delete_all=True)
            logger.info(f"Successfully deleted all vectors from index {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise

    def delete_vectors_by_ids(self, ids):
        """Delete specific vectors by their IDs"""
        try:
            index = self.pc.Index(self.index_name)
            index.delete(ids=ids)
            logger.info(f"Successfully deleted vectors with IDs: {ids}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise

# Initialize the manager
pinecone_manager = PineconeManager()
vectorstore = pinecone_manager.vectorstore

# Function to generate embeddings and store in Pinecone
def store_embeddings(chunks, vectorstore):
   try:
      vectorstore.add_documents(chunks)
      return "Embeddings stored successfully"
   except Exception as e:
      logger.error(f"Error storing embeddings: {e}")
      return "Error storing embeddings"