"""
============================================================
embeddings.py — Chunking, Embedding & Vector Store Module
============================================================
This module handles:
1. Splitting parsed documents into semantically meaningful chunks
2. Generating vector embeddings using Ollama's nomic-embed-text model
3. Creating and persisting a FAISS vector store locally
4. Loading an existing vector store on subsequent runs

The vector store is saved to data/vectorstore/ so PDFs don't need
to be re-processed every time the application starts.
============================================================
"""

import os
import logging
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from loader import load_all_documents

# ── Configure logging ──────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────
# Directory where the FAISS vector store is persisted
VECTORSTORE_DIR = os.path.join("data", "vectorstore")

# Text splitting parameters:
# - chunk_size: Maximum number of characters per chunk
#   (1000 chars ≈ ~200-250 words, good balance for retrieval)
# - chunk_overlap: Characters of overlap between consecutive chunks
#   (200 chars ensures context isn't lost at chunk boundaries)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ollama embedding model — nomic-embed-text is a high-quality
# open-source embedding model that runs locally via Ollama
# Run: ollama pull nomic-embed-text
EMBEDDING_MODEL = "nomic-embed-text"

# Ollama server URL (default local endpoint)
OLLAMA_BASE_URL = "http://localhost:11434"


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter that breaks documents into chunks.
    
    Uses RecursiveCharacterTextSplitter which tries to split on
    natural boundaries (paragraphs → sentences → words) rather
    than blindly cutting at a fixed character count. This preserves
    semantic coherence within each chunk.
    
    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Split hierarchy: try double newline first, then single, then space
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of documents into smaller chunks for embedding.
    
    Each chunk retains the metadata from its parent document,
    making it possible to trace results back to their source PDF.
    
    Args:
        documents: List of full-page Document objects from the loader.
    
    Returns:
        List of chunked Document objects, each with preserved metadata.
    """
    if not documents:
        logger.warning("No documents to chunk!")
        return []
    
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)
    
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    logger.info(f"  Chunk size: {CHUNK_SIZE} chars | Overlap: {CHUNK_OVERLAP} chars")
    
    return chunks


def get_embedding_function() -> OllamaEmbeddings:
    """
    Create an embedding function using Ollama's local embedding model.
    
    Uses nomic-embed-text which produces 768-dimensional embeddings,
    suitable for semantic similarity search. The model runs entirely
    locally via Ollama — no API keys or internet required after setup.
    
    Returns:
        OllamaEmbeddings instance configured for the embedding model.
    """
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def create_vectorstore(chunks: List[Document], batch_size: int = 50) -> FAISS:
    """
    Create a FAISS vector store from document chunks using batched embedding.
    
    Embeds chunks in small batches to avoid overwhelming Ollama's tokenizer,
    which can crash when processing too many texts in a single API call.
    
    Args:
        chunks: List of chunked Document objects.
        batch_size: Number of chunks to embed per batch (default 50).
    
    Returns:
        FAISS vector store instance ready for retrieval.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty chunk list!")
    
    logger.info(f"Creating FAISS vector store from {len(chunks)} chunks...")
    logger.info(f"  Using embedding model: {EMBEDDING_MODEL}")
    logger.info(f"  Batch size: {batch_size}")
    
    # Get the embedding function
    embeddings = get_embedding_function()
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    vectorstore = None
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.info(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        if vectorstore is None:
            # First batch: create the FAISS index
            vectorstore = FAISS.from_documents(
                documents=batch,
                embedding=embeddings,
            )
        else:
            # Subsequent batches: add to existing index
            vectorstore.add_documents(batch)
    
    # Save the vector store to disk for persistence
    save_vectorstore(vectorstore)
    
    logger.info(f"Vector store created and saved with {len(chunks)} chunks")
    return vectorstore


def save_vectorstore(vectorstore: FAISS) -> None:
    """
    Save the FAISS vector store to the local filesystem.
    
    FAISS stores are saved as a directory containing:
    - index.faiss: The actual vector index
    - index.pkl: Metadata and document mappings
    
    Args:
        vectorstore: The FAISS vector store to persist.
    """
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)
    logger.info(f"Vector store saved to: {VECTORSTORE_DIR}")


def load_vectorstore() -> Optional[FAISS]:
    """
    Load a previously saved FAISS vector store from disk.
    
    Returns:
        FAISS vector store instance, or None if no saved store exists.
    """
    # Check if the vector store directory exists and contains files
    index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    if not os.path.exists(index_path):
        logger.info("No existing vector store found on disk")
        return None
    
    try:
        logger.info(f"Loading existing vector store from: {VECTORSTORE_DIR}")
        embeddings = get_embedding_function()
        
        # Load the FAISS index with dangerous deserialization allowed
        # (required for loading pickled document metadata)
        vectorstore = FAISS.load_local(
            VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        
        logger.info("Vector store loaded successfully")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None


def get_or_create_vectorstore(force_rebuild: bool = False) -> Tuple[FAISS, dict]:
    """
    Main entry point: Get the vector store, creating it if needed.
    
    On first run (or when force_rebuild=True):
    1. Downloads and parses faculty PDFs
    2. Chunks the documents
    3. Generates embeddings and creates the FAISS index
    4. Saves everything to disk
    
    On subsequent runs:
    - Loads the existing vector store from disk (fast!)
    
    Args:
        force_rebuild: If True, rebuild even if a saved store exists.
    
    Returns:
        Tuple of (FAISS vectorstore, status dict with metadata).
        Status dict contains: 'source' ('loaded'|'created'), 'num_chunks', 'num_documents'
    """
    status = {
        "source": "unknown",
        "num_chunks": 0,
        "num_documents": 0,
        "num_pdfs": 0,
    }
    
    # Try loading existing vector store (unless force rebuild requested)
    if not force_rebuild:
        vectorstore = load_vectorstore()
        if vectorstore is not None:
            # Get the count of indexed documents from the FAISS store
            num_chunks = vectorstore.index.ntotal
            status.update({
                "source": "loaded",
                "num_chunks": num_chunks,
            })
            logger.info(f"Using existing vector store ({num_chunks} chunks indexed)")
            return vectorstore, status
    
    # Need to build the vector store from scratch
    logger.info("Building vector store from scratch...")
    
    # Step 1: Load all documents from PDFs
    documents = load_all_documents()
    
    if not documents:
        raise RuntimeError(
            "No documents were loaded! Please ensure:\n"
            "1. You have internet access (for downloading faculty PDFs), OR\n"
            "2. You've placed PDF files manually in the data/pdfs/ directory"
        )
    
    # Step 2: Chunk the documents
    chunks = chunk_documents(documents)
    
    if not chunks:
        raise RuntimeError("Document chunking produced no results!")
    
    # Step 3: Create and save the vector store
    vectorstore = create_vectorstore(chunks)
    
    # Count unique PDF sources
    unique_pdfs = set(doc.metadata.get("source", "") for doc in documents)
    
    status.update({
        "source": "created",
        "num_chunks": len(chunks),
        "num_documents": len(documents),
        "num_pdfs": len(unique_pdfs),
    })
    
    return vectorstore, status


def get_document_count() -> int:
    """
    Quick check: how many chunks are in the saved vector store?
    Returns 0 if no vector store exists.
    """
    index_path = os.path.join(VECTORSTORE_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return 0
    
    try:
        vectorstore = load_vectorstore()
        if vectorstore:
            return vectorstore.index.ntotal
    except Exception:
        pass
    
    return 0


# ── CLI testing ────────────────────────────────────────────
if __name__ == "__main__":
    """Quick test: build/load the vector store and print stats."""
    vectorstore, status = get_or_create_vectorstore()
    print(f"\nVector Store Status:")
    print(f"  Source: {status['source']}")
    print(f"  Chunks indexed: {status['num_chunks']}")
    print(f"  Documents: {status['num_documents']}")
    print(f"  PDFs: {status['num_pdfs']}")
