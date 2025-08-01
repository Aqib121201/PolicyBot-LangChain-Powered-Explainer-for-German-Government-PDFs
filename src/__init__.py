"""
PolicyBot: LangChain-Powered Q&A and Summarizer for German Government PDFs

A comprehensive system for processing German government documents with
intelligent question answering, summarization, and analysis capabilities.
"""

__version__ = "1.0.0"
__author__ = "PolicyBot Team"
__email__ = "policybot@example.com"

from .config import Config
from .pdf_processor import PDFProcessor
from .langchain_qa import LangChainQA
from .summarizer import DocumentSummarizer
from .keyword_extractor import KeywordExtractor
from .vector_store import VectorStore

__all__ = [
    "Config",
    "PDFProcessor", 
    "LangChainQA",
    "DocumentSummarizer",
    "KeywordExtractor",
    "VectorStore"
] 