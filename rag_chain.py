"""
============================================================
rag_chain.py — LangChain RAG Chain Module
============================================================
This module builds the Retrieval-Augmented Generation chain:
1. Connects the FAISS vector store retriever to an Ollama LLM
2. Uses a custom prompt template to ground answers in context
3. Provides model selection and Ollama connectivity checks

The chain flow:
  User Question → Retriever (FAISS) → Top-K Chunks → 
  Prompt Template (question + context) → Ollama LLM → Answer
============================================================
"""

import logging
from typing import Dict, Optional, List

import requests
from langchain_ollama import ChatOllama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

# ── Configure logging ──────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────
# Ollama server URL (default local endpoint)
OLLAMA_BASE_URL = "http://localhost:11434"

# Default LLM model to use for answer generation
DEFAULT_MODEL = "llama3"

# Number of relevant chunks to retrieve for each question
# 4 provides good context coverage without overwhelming the LLM
RETRIEVER_K = 4

# ── Custom RAG Prompt Template ─────────────────────────────
# This prompt instructs the LLM to:
# 1. Only use the provided context to answer
# 2. Acknowledge when information isn't available
# 3. Be specific and cite faculty names/details
RAG_PROMPT_TEMPLATE = """You are a knowledgeable assistant for the University Institute of Engineering and Technology (UIET), Panjab University, Chandigarh. Your role is to answer questions about faculty members based on the provided context.

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context below.
- If the context doesn't contain enough information to answer the question, clearly state: "I don't have enough information in the available documents to answer this question."
- Be specific — mention faculty names, designations, departments, qualifications, and other details when available.
- Format your response clearly with proper structure when listing multiple items.
- Do NOT make up or assume any information not present in the context.

CONTEXT FROM FACULTY DOCUMENTS:
{context}

QUESTION: {question}

ANSWER:"""


def get_rag_prompt() -> PromptTemplate:
    """
    Create the custom prompt template for the RAG chain.
    
    The template has two input variables:
    - context: Retrieved document chunks relevant to the question
    - question: The user's natural language question
    
    Returns:
        PromptTemplate configured for the RAG pipeline.
    """
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )


def get_llm(model_name: str = DEFAULT_MODEL) -> ChatOllama:
    """
    Initialize an Ollama LLM instance.
    
    Uses ChatOllama for chat-optimized responses (vs plain Ollama
    which is for completion-style models).
    
    Args:
        model_name: Name of the Ollama model to use (e.g., 'llama3', 'mistral').
    
    Returns:
        Configured ChatOllama instance.
    """
    return ChatOllama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,  # Low temperature for factual, consistent answers
    )


def build_rag_chain(
    vectorstore: FAISS,
    model_name: str = DEFAULT_MODEL,
) -> RetrievalQA:
    """
    Build the complete RAG chain connecting retriever → prompt → LLM.
    
    The chain works as follows:
    1. User asks a question
    2. The retriever searches FAISS for the top-K most similar chunks
    3. Retrieved chunks are inserted into the prompt template as context
    4. The Ollama LLM generates an answer grounded in that context
    
    Args:
        vectorstore: FAISS vector store with indexed faculty documents.
        model_name: Ollama model to use for generation.
    
    Returns:
        RetrievalQA chain ready to answer questions.
    """
    logger.info(f"Building RAG chain with model: {model_name}")
    
    # Create the retriever from the vector store
    # search_kwargs={'k': 4} means retrieve the 4 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    
    # Initialize the Ollama LLM
    llm = get_llm(model_name)
    
    # Get the custom prompt template
    prompt = get_rag_prompt()
    
    # Build the RetrievalQA chain
    # chain_type="stuff" means all retrieved chunks are "stuffed" into
    # a single prompt (works well when chunks are small enough)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,  # Return source docs with the answer
        chain_type_kwargs={"prompt": prompt},
    )
    
    logger.info("RAG chain built successfully")
    return chain


def query_chain(chain: RetrievalQA, question: str) -> Dict:
    """
    Send a question to the RAG chain and get the response.
    
    Args:
        chain: The built RetrievalQA chain.
        question: User's natural language question.
    
    Returns:
        Dictionary with:
        - 'answer': The LLM's response text
        - 'source_documents': List of source Document objects used
    """
    try:
        logger.info(f"Processing question: {question[:80]}...")
        
        # Invoke the chain
        result = chain.invoke({"query": question})
        
        # Extract answer and source documents
        answer = result.get("result", "No answer generated.")
        source_docs = result.get("source_documents", [])
        
        logger.info(f"Answer generated ({len(answer)} chars) from {len(source_docs)} sources")
        
        return {
            "answer": answer,
            "source_documents": source_docs,
        }
        
    except Exception as e:
        logger.error(f"Error querying chain: {e}")
        return {
            "answer": f"Error generating response: {str(e)}",
            "source_documents": [],
        }


def check_ollama_status() -> Dict:
    """
    Check if Ollama is running and which models are available.
    
    Pings the Ollama API endpoint to verify connectivity,
    then lists available models.
    
    Returns:
        Dictionary with:
        - 'running': bool — whether Ollama is accessible
        - 'models': list — names of available models
        - 'error': str — error message if not running
    """
    try:
        # Check if Ollama API is responsive
        response = requests.get(
            f"{OLLAMA_BASE_URL}/api/tags",
            timeout=5,
        )
        response.raise_for_status()
        
        # Parse available models
        data = response.json()
        models = [m["name"] for m in data.get("models", [])]
        
        return {
            "running": True,
            "models": models,
            "error": None,
        }
        
    except requests.exceptions.ConnectionError:
        return {
            "running": False,
            "models": [],
            "error": "Cannot connect to Ollama. Is it running? Start with: ollama serve",
        }
    except Exception as e:
        return {
            "running": False,
            "models": [],
            "error": f"Ollama error: {str(e)}",
        }


def get_available_models() -> List[str]:
    """
    Get a list of available Ollama model names.
    Returns a fallback list if Ollama is unreachable.
    """
    status = check_ollama_status()
    if status["running"] and status["models"]:
        return status["models"]
    # Fallback: common model names
    return ["llama3", "mistral", "llama3.1", "gemma2", "phi3"]


# ── CLI testing ────────────────────────────────────────────
if __name__ == "__main__":
    """Quick test: check Ollama status and print available models."""
    status = check_ollama_status()
    print(f"\nOllama Status:")
    print(f"  Running: {status['running']}")
    print(f"  Models: {status['models']}")
    if status['error']:
        print(f"  Error: {status['error']}")
