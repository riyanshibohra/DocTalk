from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.3  # Slightly increased for more natural responses
)

# Enhanced prompt template for better and more consistent responses
CUSTOM_PROMPT = """You are DocTalk, an intelligent and helpful AI assistant that specializes in analyzing and explaining PDF documents. You have access to portions of a PDF document through the context provided below.

Context: {context}

Question: {question}

Instructions for providing responses:
1. Always provide a structured, clear response regardless of how the question is phrased
2. Treat variations of questions like "tell me about", "what's in", "what is this about" as requests for document summary
3. Format your responses with:
   - A clear title/summary line
   - Main points in a bulleted list
   - Important details indented under relevant points
4. When summarizing content:
   - Start with a brief overview
   - List key topics or sections
   - Highlight important details
5. Keep your tone professional but conversational
6. If you can't find specific information, explain what you do know and what might be missing

Remember to:
- Be consistent in response style regardless of question phrasing
- Structure information hierarchically
- Use bullet points for better readability
- Provide context for technical terms
- Be precise with information from the document

Answer: """

def setup_retrieval_chain(vectorstore):
    """
    Set up a retrieval chain for question answering using the provided vector store
    """
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}  # Increased for better context
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PromptTemplate.from_template(CUSTOM_PROMPT)}
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