from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

# Custom prompt template for better context handling and inference
CUSTOM_PROMPT = """You are a helpful AI assistant that analyzes and explains documents. You have access to portions of a PDF document through the context provided below.

Context: {context}

Question: {question}

Instructions:
1. Use the provided context to understand the document's content
2. When asked about key points, takeaways, or summaries:
   - Analyze the available information
   - Synthesize the main ideas and concepts
   - Present them in a clear, structured format (preferably bullet points)
3. Draw reasonable conclusions from the context even if not explicitly stated
4. If the context is insufficient, explain what aspects you can understand and what's missing
5. Always maintain accuracy while providing comprehensive answers

Answer: """

def setup_retrieval_chain(vectorstore):
    """
    Set up a retrieval chain for question answering using the provided vector store
    """
    try:
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=CUSTOM_PROMPT
        )

        # Create retrieval chain with simpler search parameters
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 5}  # Just use k parameter
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": prompt,
                "document_separator": "\n\n"
            }
        )
        
        logger.info("Retrieval chain setup successfully")
        return qa_chain
    
    except Exception as e:
        logger.error(f"Error setting up retrieval chain: {e}")
        raise

def get_answer(qa_chain, question, chat_history=[]):
    """
    Get answer for a question using the retrieval chain
    """
    try:
        # Get response from the chain
        response = qa_chain({"question": question, "chat_history": chat_history})
        
        # Extract answer and source documents
        answer = response["answer"]
        source_docs = response["source_documents"]
        
        # Log sources used for the answer
        sources = [doc.metadata.get('source', 'Unknown') for doc in source_docs]
        logger.info(f"Sources used for answer: {sources}")
        
        return answer, source_docs
    
    except Exception as e:
        logger.error(f"Error getting answer: {e}")
        return f"Error: {str(e)}", [] 