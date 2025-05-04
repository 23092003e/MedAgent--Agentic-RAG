import logging
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from ..config import LLM_MODEL_NAME, LLM_TEMPERATURE, RETRIEVAL_K

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def set_custom_prompt() -> PromptTemplate:
    template = """
You are a helpful and knowledgeable medical assistant. Use only the information from the context below to answer the user's question.

Do not use prior knowledge.

If the answer cannot be found in the context, respond with "I'm not sure based on the provided information."

Always include a brief explanation and cite the source document if relevant.

Context:
{context}

Question:
{question}

Answer:
"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

def load_llm() -> Ollama:
    return Ollama(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)

def create_qa_chain(vectorstore) -> RetrievalQA:
    prompt = set_custom_prompt()
    llm = load_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    logger.info("QA chain created with model %s using 'stuff' chain type", LLM_MODEL_NAME)
    return qa
