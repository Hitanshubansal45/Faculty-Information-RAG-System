"""
============================================================
loader.py — PDF Downloading & Parsing Module
============================================================
This module handles:
1. Scraping UIET Panjab University department pages for PDF links
2. Downloading faculty profile PDFs to a local directory
3. Parsing PDF content into LangChain Document objects

The hybrid approach ensures reliability:
  - Scrapes department pages for .pdf links
  - Falls back to a hardcoded list of known faculty profile URLs
  - Supports manually placed PDFs in the data/pdfs/ directory
============================================================
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pdfplumber
from langchain_core.documents import Document

# ── Configure logging ──────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────
# Directory where downloaded PDFs are stored
PDF_DIR = os.path.join("data", "pdfs")

# Request timeout in seconds
REQUEST_TIMEOUT = 30

# Headers to mimic a real browser (some university sites block raw requests)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── UIET Department Pages to Scrape for PDF Links ─────────
# These are the actual working department faculty listing pages on the UIET website (using page_id).
DEPARTMENT_PAGES = [
    "https://uiet.puchd.ac.in/?page_id=57",   # CSE Faculty
    "https://uiet.puchd.ac.in/?page_id=21",   # IT Faculty
    "https://uiet.puchd.ac.in/?page_id=222",  # ECE Faculty
    "https://uiet.puchd.ac.in/?page_id=905",  # EEE Faculty
    "https://uiet.puchd.ac.in/?page_id=423",  # BIO Faculty
    "https://uiet.puchd.ac.in/?page_id=450",  # Mechanical Faculty
    "https://uiet.puchd.ac.in/?page_id=484",  # Applied Sciences Faculty
]

# ── Fallback: Known Faculty Profile PDF URLs ──────────────
# These are verified PDF URLs for faculty profiles.
KNOWN_FACULTY_PDFS = [
    "https://uiet.puchd.ac.in/resumes/profsarabjeetcse.pdf",
    "https://uiet.puchd.ac.in/wp-content/uploads/2021/02/ProfSavita2021.pdf",
    "https://uiet.puchd.ac.in/wp-content/uploads/2025/05/drkrishankumar.pdf",
    "https://uiet.puchd.ac.in/wp-content/uploads/2023/04/ProfRenuVig.pdf",
    "https://uiet.puchd.ac.in/wp-content/uploads/FacultyCV/Mechanical/Surjeet-Singh.pdf",
    "https://uiet.puchd.ac.in/wp-content/uploads/2025/05/dranilkumarapp.pdf"
]


def ensure_pdf_directory() -> str:
    """
    Create the PDF storage directory if it doesn't exist.
    
    Returns:
        str: Absolute path to the PDF directory.
    """
    os.makedirs(PDF_DIR, exist_ok=True)
    return PDF_DIR


def scrape_pdf_links_from_page(url: str) -> List[str]:
    """
    Scrape a single webpage for links that point to PDF files.
    
    Args:
        url: The webpage URL to scrape.
    
    Returns:
        List of absolute PDF URLs found on the page.
    """
    pdf_links = []
    try:
        logger.info(f"Scraping page for PDF links: {url}")
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Find all anchor tags with href attributes
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # Check if the link points to a PDF file
            if href.lower().endswith(".pdf"):
                # Convert relative URLs to absolute
                absolute_url = urljoin(url, href)
                pdf_links.append(absolute_url)
                logger.info(f"  Found PDF: {absolute_url}")
                
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to scrape {url}: {e}")
    except Exception as e:
        logger.warning(f"Error parsing {url}: {e}")
    
    return pdf_links


def discover_pdf_urls() -> List[str]:
    """
    Discover all available faculty PDF URLs using a hybrid approach:
    1. Scrape UIET department pages for PDF links
    2. Add known fallback PDF URLs
    3. Deduplicate the combined list
    
    Returns:
        List of unique PDF URLs to download.
    """
    all_pdf_urls = set()
    
    # Step 1: Scrape department pages for PDF links
    logger.info("=" * 60)
    logger.info("Phase 1: Scraping department pages for PDF links...")
    logger.info("=" * 60)
    for page_url in DEPARTMENT_PAGES:
        found_links = scrape_pdf_links_from_page(page_url)
        all_pdf_urls.update(found_links)
    
    # Step 2: Add known fallback URLs
    logger.info("=" * 60)
    logger.info("Phase 2: Adding known fallback faculty PDF URLs...")
    logger.info("=" * 60)
    for pdf_url in KNOWN_FACULTY_PDFS:
        all_pdf_urls.add(pdf_url)
        logger.info(f"  Added fallback: {pdf_url}")
    
    logger.info(f"\nTotal unique PDF URLs discovered: {len(all_pdf_urls)}")
    return list(all_pdf_urls)


def download_pdf(url: str, save_dir: str) -> Optional[str]:
    """
    Download a single PDF file to the specified directory.
    Skips download if the file already exists locally.
    
    Args:
        url: The URL of the PDF to download.
        save_dir: Directory to save the downloaded PDF.
    
    Returns:
        Path to the downloaded file, or None if download failed.
    """
    # Generate a clean filename from the URL
    parsed_url = urlparse(url)
    # Use the last part of the URL path as filename
    filename = os.path.basename(parsed_url.path)
    
    # If filename is empty or doesn't end with .pdf, create one from URL hash
    if not filename or not filename.lower().endswith(".pdf"):
        # Create a safe filename from the full URL
        safe_name = re.sub(r'[^\w\-.]', '_', parsed_url.path.strip('/'))
        filename = f"{safe_name}.pdf"
    
    filepath = os.path.join(save_dir, filename)
    
    # Skip download if file already exists (avoid re-downloading)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        logger.info(f"  Already exists, skipping: {filename}")
        return filepath
    
    try:
        logger.info(f"  Downloading: {filename}")
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, stream=True)
        response.raise_for_status()
        
        # Verify it's actually a PDF (check Content-Type header)
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" not in content_type and "octet-stream" not in content_type:
            logger.warning(f"  Skipping {filename}: Content-Type is '{content_type}', not a PDF")
            return None
        
        # Save the PDF file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"  Saved: {filepath} ({os.path.getsize(filepath)} bytes)")
        return filepath
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"  Failed to download {url}: {e}")
        return None


def download_all_pdfs() -> List[str]:
    """
    Discover and download all faculty PDFs.
    
    Returns:
        List of file paths to successfully downloaded PDFs.
    """
    ensure_pdf_directory()
    
    # Discover PDF URLs from scraping + fallback
    pdf_urls = discover_pdf_urls()
    
    if not pdf_urls:
        logger.warning("No PDF URLs discovered. Place PDFs manually in data/pdfs/")
        # Still check for manually placed PDFs
        return get_local_pdf_paths()
    
    # Download each PDF
    logger.info("=" * 60)
    logger.info("Phase 3: Downloading PDFs...")
    logger.info("=" * 60)
    downloaded_paths = []
    for url in pdf_urls:
        result = download_pdf(url, PDF_DIR)
        if result:
            downloaded_paths.append(result)
    
    # Also include any manually placed PDFs that weren't downloaded
    all_local_pdfs = get_local_pdf_paths()
    unique_paths = list(set(downloaded_paths + all_local_pdfs))
    
    logger.info(f"\nTotal PDFs available: {len(unique_paths)}")
    return unique_paths


def get_local_pdf_paths() -> List[str]:
    """
    Get paths to all PDF files in the local data/pdfs/ directory.
    Useful for including manually placed PDFs.
    
    Returns:
        List of file paths to local PDF files.
    """
    ensure_pdf_directory()
    pdf_paths = []
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_paths.append(os.path.join(PDF_DIR, filename))
    return pdf_paths


def parse_pdf_to_documents(pdf_path: str) -> List[Document]:
    """
    Parse a single PDF file and convert it to LangChain Document objects.
    Each page becomes a separate Document with metadata.
    
    Uses pdfplumber for robust text extraction (handles tables, 
    multi-column layouts, and complex formatting better than PyPDF2).
    
    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        List of Document objects (one per page with content).
    """
    documents = []
    filename = os.path.basename(pdf_path)
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text from the page
                text = page.extract_text()
                
                # Skip empty pages
                if not text or text.strip() == "":
                    continue
                
                # Clean up the extracted text
                # Remove excessive whitespace while preserving paragraph breaks
                text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
                text = re.sub(r' {2,}', ' ', text)       # Collapse multiple spaces
                text = text.strip()
                
                # Create a LangChain Document with rich metadata
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "total_pages": len(pdf.pages),
                        "file_path": pdf_path,
                    }
                )
                documents.append(doc)
                
        logger.info(f"  Parsed {len(documents)} pages from: {filename}")
        
    except Exception as e:
        logger.error(f"  Failed to parse {pdf_path}: {e}")
    
    return documents


def load_all_documents() -> List[Document]:
    """
    Main entry point: Download all faculty PDFs and parse them into Documents.
    
    This function orchestrates the full pipeline:
    1. Discover PDF URLs (scraping + fallback)
    2. Download PDFs to local directory
    3. Parse each PDF into LangChain Documents
    
    Returns:
        List of all Document objects from all successfully parsed PDFs.
    """
    logger.info("=" * 60)
    logger.info("STARTING DOCUMENT LOADING PIPELINE")
    logger.info("=" * 60)
    
    # Step 1 & 2: Discover and download PDFs
    pdf_paths = download_all_pdfs()
    
    if not pdf_paths:
        logger.warning("No PDFs found! Please place PDF files in the data/pdfs/ directory.")
        return []
    
    # Step 3: Parse each PDF into Documents
    logger.info("=" * 60)
    logger.info("Phase 4: Parsing PDFs into documents...")
    logger.info("=" * 60)
    all_documents = []
    for pdf_path in pdf_paths:
        docs = parse_pdf_to_documents(pdf_path)
        all_documents.extend(docs)
    
    logger.info("=" * 60)
    logger.info(f"LOADING COMPLETE: {len(all_documents)} document pages from {len(pdf_paths)} PDFs")
    logger.info("=" * 60)
    
    return all_documents


# ── CLI testing ────────────────────────────────────────────
if __name__ == "__main__":
    """Quick test: run this file directly to test PDF downloading and parsing."""
    docs = load_all_documents()
    print(f"\nLoaded {len(docs)} document pages total.")
    if docs:
        print(f"\nSample document (first 300 chars):")
        print(f"  Source: {docs[0].metadata['source']}")
        print(f"  Content: {docs[0].page_content[:300]}...")
