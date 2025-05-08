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
4. Provide clear explanations with relevant citations
5. Use professional medical terminology when appropriate
6. Include important disclaimers when necessary

Here are some examples of good responses:

Question: What are the symptoms of diabetes?
Context: According to medical literature, common diabetes symptoms include increased thirst, frequent urination, unexplained weight loss, and fatigue. Type 2 diabetes may develop gradually.
Answer: Based on the provided information, the main symptoms of diabetes include:
- Increased thirst
- Frequent urination
- Unexplained weight loss
- Fatigue
It's important to note that Type 2 diabetes symptoms may develop gradually. Please consult a healthcare provider for proper diagnosis and treatment.

Question: How is high blood pressure treated?
Context: Treatment options for hypertension include lifestyle modifications (reduced salt intake, regular exercise) and medications like ACE inhibitors and beta blockers.
Answer: According to the context, high blood pressure can be managed through:
1. Lifestyle modifications:
   - Reducing salt intake
   - Regular exercise
2. Medications:
   - ACE inhibitors
   - Beta blockers
Please consult your healthcare provider for personalized treatment recommendations.

Current Question: {question}

Relevant Context:
{context}

Answer:"""

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
    """
    Create a QA chain with improved retrieval and error handling
    
    Args:
        vectorstore: Vector store instance
        
    Returns:
        RetrievalQA chain
    """
    try:
        prompt = set_custom_prompt()
        llm = load_llm()
        
        # Configure retriever with search parameters
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": RETRIEVAL_K,
                "fetch_k": RETRIEVAL_K * 2  # Fetch more candidates for better selection
            }
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # Use stuff chain for simplicity and better context handling
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "verbose": True  # Enable verbose mode for better debugging
            }
        )

        logger.info(
            "QA chain created with model %s using 'stuff' chain type and k=%d", 
            LLM_MODEL_NAME, 
            RETRIEVAL_K
        )
        return qa
        
    except Exception as e:
        logger.error(f"Failed to create QA chain: {str(e)}")
        raise
