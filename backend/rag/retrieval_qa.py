import logging
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from backend.config import LLM_MODEL_NAME, LLM_TEMPERATURE, RETRIEVAL_K

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_custom_prompt() -> PromptTemplate:
    template = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say you don't know.
Context: {context}
Question: {question}
"""
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def load_llm() -> Ollama:
    return Ollama(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE)


def create_qa_chain(vectorstore) -> RetrievalQA:
    prompt = set_custom_prompt()
    llm = load_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    logger.info("QA chain created with model %s", LLM_MODEL_NAME)
    return qa