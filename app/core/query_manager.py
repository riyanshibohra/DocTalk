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

# Custom prompt template for better context handling
CUSTOM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context from documents.
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer: """

def setup_retrieval_chain(vectorstore):
    """
    Set up a retrieval chain for question answering using the provided vector store
    """
    try:
        # Create prompt template with more explicit instructions
        CUSTOM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context from PDF documents.
        
        Context: {context}
        
        Question: {question}
        
        Instructions:
        1. Use ONLY the information from the context above to answer the question
        2. If you can't find relevant information in the context, say so
        3. Be specific and cite information directly from the context
        4. If asked for key points or takeaways, structure your response in bullet points
        
        Answer: """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=CUSTOM_PROMPT
        )

        # Create retrieval chain with modified parameters
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt}
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