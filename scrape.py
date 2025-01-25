# Importing the necessary libraries
import os
import requests
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import re
import warnings
warnings.filterwarnings("ignore")


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# Initialize Pinecone client
pc = PineconeClient()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create or get index
index_name = "voice-assistant"

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
vectorstore = Pinecone(
    index=pc.Index(index_name),
    embedding=embeddings,
    text_key="text"
)

# Function to get the documentation URLs

def get_documentation_urls():

    return[
        	'/docs/huggingface_hub/guides/overview',
		    '/docs/huggingface_hub/guides/download',
		    '/docs/huggingface_hub/guides/upload',
		    '/docs/huggingface_hub/guides/hf_file_system',
		    '/docs/huggingface_hub/guides/repository',
		    '/docs/huggingface_hub/guides/search',
    ]

# Function to construct the full URL

def construct_full_url(base_url, relative_url):
    return base_url + relative_url

def scrape_page_content(url):
    response = requests.get(url)    #get request to the url
    soup = BeautifulSoup(response.text, 'html.parser')    #use beautiful soup to parse the html content
    text = soup.body.text.strip()     # Extract the desired content from the page (in this case, the body text)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = re.sub(r'\s+', ' ', text)    # Remove any whitespace characters
    return text.strip()

# Function to scrape all content from the given URLs and save to a file

def scrape_all_content(base_url,relative_urls,filename):

    content = []
    for i in relative_urls:
        full_url = construct_full_url(base_url,i)
        scraped_content = scrape_page_content(full_url)
        content.append(scraped_content.rstrip('\n'))

    # Save the content to a file
    with open(filename, 'w', encoding='utf-8') as file:
        for item in content:
            file.write("%s\n" % item)
    return content

# Define a function to load documents from a file

def load_docs(root_dir,filename):
    docs = []
    try:
        loader = TextLoader(os.path.join(root_dir,filename), encoding='utf-8')
        docs.extend(loader.load_and_split())
    except Exception as e:
        pass      #if an error occurs, pass it and continue
    return docs

def split_docs(docs):
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap = 0)
    return text_splitter.split_documents(docs)

# define the main function

def main():
    base_url = 'https://huggingface.co'
    filename='content.txt'
    root_dir ='./'
    relative_urls = get_documentation_urls()

    content = scrape_all_content(base_url,relative_urls,filename)
    docs = load_docs(root_dir,filename)
    texts = split_docs(docs)

    # Extract text content and create IDs
    text_contents = [doc.page_content for doc in texts]
    doc_ids = [f"doc_{i}" for i in range(len(texts))]

    # Add documents to vector store with IDs
    vectorstore.add_texts(
        texts=text_contents,
        ids=doc_ids,
        metadatas=[doc.metadata for doc in texts]
    )

    os.remove(filename)
    print("Content scraped, embedded, and stored in Pinecone successfully!")

if __name__ == "__main__":
    main()