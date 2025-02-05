from langchain_openai import OpenAIEmbeddings
#from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import logging
import time

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
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        # Create index if it doesn't exist
        self._create_index_if_not_exists()
        self.vectorstore = self.initialize_pinecone_index()

    def _create_index_if_not_exists(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(5)
            else:
                logger.info(f"Index {self.index_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    def initialize_pinecone_index(self):
        """Initialize Pinecone vector store"""
        try:
            return PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=""  # Add default namespace
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise

    def delete_all_vectors(self):
        """Delete all vectors from the index"""
        try:
            index = self.pc.Index(self.index_name)
            index.delete(delete_all=True, namespace="")  # Specify namespace
            logger.info(f"Successfully deleted all vectors from index {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False  # Return False instead of raising

    def delete_vectors_by_ids(self, ids):
        """Delete specific vectors by their IDs"""
        try:
            index = self.pc.Index(self.index_name)
            index.delete(ids=ids, namespace="")  # Specify namespace
            logger.info(f"Successfully deleted vectors with IDs: {ids}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False  # Return False instead of raising

# Function to generate embeddings and store in Pinecone
def store_embeddings(chunks, vectorstore):
   try:
      vectorstore.add_documents(chunks)
      logger.info(f"Stored {len(chunks)} embeddings in Pinecone")  # Log the number of stored embeddings
      return "Embeddings stored successfully"
   except Exception as e:
      logger.error(f"Error storing embeddings: {e}")
      return "Error storing embeddings"