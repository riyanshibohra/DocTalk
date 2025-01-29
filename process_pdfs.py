import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Function to chunk text
def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# Function to generate embeddings and store in Pinecone
def store_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    index = pinecone.Index('your-index-name')
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed(chunk)
        index.upsert([(f'chunk-{i}', embedding)])

# Example usage
if __name__ == "__main__":
    pdf_path = 'sample.pdf'  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(extracted_text)
    store_embeddings(chunks) 