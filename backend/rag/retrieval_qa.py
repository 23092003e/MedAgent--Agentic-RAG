import logging
from typing import Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from ..config import LLM_MODEL_NAME, LLM_TEMPERATURE, RETRIEVAL_K

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def set_custom_prompt() -> PromptTemplate:
    """Create a custom prompt template with few-shot examples"""
    
    template = """You are a knowledgeable and professional medical assistant. Your role is to provide accurate, evidence-based medical information using only the provided context. Follow these guidelines:

1. Use ONLY information from the provided context
2. If the context doesn't contain enough information, say "I cannot provide a complete answer based on the available information"
3. Avoid making assumptions or using external knowledge
4. For each claim or piece of information, cite the specific source document and page number in [square brackets]
5. Use professional medical terminology when appropriate
6. Include important disclaimers when necessary

Here are some examples of good responses:

Question: What are the symptoms of diabetes?
Context: [From diabetes_guide.pdf, page 12] Common diabetes symptoms include increased thirst, frequent urination, unexplained weight loss, and fatigue. Type 2 diabetes may develop gradually.
Answer: Based on the provided information [diabetes_guide.pdf, p.12], the main symptoms of diabetes include:
- Increased thirst
- Frequent urination
- Unexplained weight loss
- Fatigue
It's important to note that Type 2 diabetes symptoms may develop gradually. Please consult a healthcare provider for proper diagnosis and treatment.

Current Question: {question}

Relevant Context:
{context}

Answer: Please provide your response with clear references to the source documents in [square brackets] after each claim.
"""

    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

def load_llm() -> Ollama:
    """Load and configure the LLM"""
    try:
        return Ollama(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE
        )
    except Exception as e:
        logger.error(f"Failed to load LLM: {str(e)}")
        raise

def create_qa_chain(vectorstore) -> RetrievalQA:
    """Create a QA chain with improved retrieval and error handling"""
    try:
        prompt = set_custom_prompt()
        llm = load_llm()
        
        # Configure retriever with better search parameters
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_K * 2,
                "score_threshold": 0.3  # Lower threshold for better recall
            }
        )

        # Create chain with better error handling
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True
            }
        )

        # Create wrapper for error handling
        def qa_with_fallback(query: Dict) -> Dict:
            try:
                # First try direct chain call
                result = qa(query)
                logger.info(f"Retrieved {len(result.get('source_documents', []))} documents")
                
                # Log source content for debugging
                for i, doc in enumerate(result.get('source_documents', [])):
                    logger.debug(f"Document {i+1} content: {doc.page_content[:200]}...")
                
                # Check if result is empty or too short
                if not result.get('result') or len(result['result'].strip()) < 10:
                    # Try with lower threshold
                    retriever.search_kwargs["score_threshold"] = 0.1
                    result = qa(query)
                    retriever.search_kwargs["score_threshold"] = 0.3  # Reset threshold
                    
                    if not result.get('result') or len(result['result'].strip()) < 10:
                        return {
                            'result': "I apologize, but I couldn't find enough relevant information to answer your question accurately. Could you please:\n" +
                                     "1. Rephrase your question\n" +
                                     "2. Be more specific\n" +
                                     "3. Or ask about a different medical topic",
                            'source_documents': result.get('source_documents', [])
                        }
                return result
                
            except Exception as e:
                logger.error(f"QA chain error: {str(e)}")
                return {
                    'result': "I encountered an error while processing your question. Please try asking in a different way.",
                    'source_documents': []
                }

        # Create a callable class to wrap the chain
        class QAWithFallback:
            def __init__(self, qa_chain, fallback_fn):
                self.qa_chain = qa_chain
                self.fallback_fn = fallback_fn
                
            def __call__(self, query):
                logger.info(f"Processing query: {query.get('query', '')}")
                return self.fallback_fn(query)

        # Return wrapped chain
        return QAWithFallback(qa, qa_with_fallback)

    except Exception as e:
        logger.error(f"Failed to create QA chain: {str(e)}")
        raise
