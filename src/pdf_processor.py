"""
PDF Processor for German Government Documents.

Handles PDF text extraction, OCR processing, and text preprocessing
for German government documents from govdata.de.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import numpy as np
import pandas as pd
from loguru import logger

from .config import get_config


@dataclass
class DocumentMetadata:
    """Metadata extracted from PDF documents."""
    
    title: str
    author: str
    subject: str
    creator: str
    producer: str
    creation_date: str
    modification_date: str
    num_pages: int
    file_size: int
    language: str = "de"
    document_type: str = "government"
    source: str = "govdata.de"


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    
    text: str
    page_number: int
    chunk_id: str
    start_char: int
    end_char: int
    confidence: float = 1.0
    language: str = "de"


class PDFProcessor:
    """Handles PDF processing for German government documents."""
    
    def __init__(self, config=None):
        """Initialize PDF processor with configuration."""
        self.config = config or get_config()
        self.logger = logger.bind(name="PDFProcessor")
        
        # German-specific text patterns
        self.german_patterns = {
            'umlauts': r'[äöüÄÖÜß]',
            'german_words': r'\b[A-Za-zäöüÄÖÜß]+\b',
            'german_sentences': r'[^.!?]*[.!?]',
            'page_numbers': r'^\s*\d+\s*$',
            'headers': r'^(Bundesregierung|Bundesministerium|Bundestag|Bundesrat)',
            'footers': r'(Seite|Page)\s+\d+',
        }
        
        # Initialize OCR if enabled
        if self.config.processing.ocr_enabled:
            try:
                pytesseract.get_tesseract_version()
                self.logger.info("Tesseract OCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Tesseract OCR not available: {e}")
                self.config.processing.ocr_enabled = False
    
    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a PDF file and extract text, metadata, and chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and chunks
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Processing PDF: {pdf_path.name}")
        
        # Extract metadata
        metadata = self._extract_metadata(pdf_path)
        
        # Extract text
        text_content = self._extract_text(pdf_path)
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(text_content)
        
        # Create text chunks
        chunks = self._create_chunks(cleaned_text)
        
        # Analyze document structure
        structure = self._analyze_structure(cleaned_text, chunks)
        
        return {
            'metadata': metadata,
            'raw_text': text_content,
            'cleaned_text': cleaned_text,
            'chunks': chunks,
            'structure': structure,
            'file_path': str(pdf_path)
        }
    
    def _extract_metadata(self, pdf_path: Path) -> DocumentMetadata:
        """Extract metadata from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata
                
                metadata = DocumentMetadata(
                    title=info.get('/Title', pdf_path.stem),
                    author=info.get('/Author', 'Unknown'),
                    subject=info.get('/Subject', ''),
                    creator=info.get('/Creator', ''),
                    producer=info.get('/Producer', ''),
                    creation_date=info.get('/CreationDate', ''),
                    modification_date=info.get('/ModDate', ''),
                    num_pages=len(pdf_reader.pages),
                    file_size=pdf_path.stat().st_size,
                    language=self._detect_language(pdf_path)
                )
                
                self.logger.info(f"Extracted metadata: {metadata.title}")
                return metadata
                
        except Exception as e:
            self.logger.error(f"Error extracting metadata: {e}")
            return DocumentMetadata(
                title=pdf_path.stem,
                author="Unknown",
                subject="",
                creator="",
                producer="",
                creation_date="",
                modification_date="",
                num_pages=0,
                file_size=pdf_path.stat().st_size
            )
    
    def _extract_text(self, pdf_path: Path) -> str:
        """Extract text from PDF using multiple methods."""
        text_content = ""
        
        # Try pdfplumber first (better for structured documents)
        try:
            text_content = self._extract_with_pdfplumber(pdf_path)
            if text_content.strip():
                self.logger.info("Text extracted successfully with pdfplumber")
                return text_content
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Fallback to PyPDF2
        try:
            text_content = self._extract_with_pypdf2(pdf_path)
            if text_content.strip():
                self.logger.info("Text extracted successfully with PyPDF2")
                return text_content
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Try OCR if enabled
        if self.config.processing.ocr_enabled:
            try:
                text_content = self._extract_with_ocr(pdf_path)
                if text_content.strip():
                    self.logger.info("Text extracted successfully with OCR")
                    return text_content
            except Exception as e:
                self.logger.error(f"OCR extraction failed: {e}")
        
        raise ValueError("Could not extract text from PDF using any method")
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Seite {page_num} ---\n{page_text}")
                except Exception as e:
                    self.logger.warning(f"Error extracting text from page {page_num}: {e}")
        
        return "\n\n".join(text_parts)
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> str:
        """Extract text using PyPDF2."""
        text_parts = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Seite {page_num} ---\n{page_text}")
                except Exception as e:
                    self.logger.warning(f"Error extracting text from page {page_num}: {e}")
        
        return "\n\n".join(text_parts)
    
    def _extract_with_ocr(self, pdf_path: Path) -> str:
        """Extract text using OCR (Tesseract)."""
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    # Convert page to image
                    img = page.to_image()
                    
                    # Perform OCR
                    ocr_text = pytesseract.image_to_string(
                        img.original,
                        lang=self.config.processing.ocr_language,
                        config='--psm 6'
                    )
                    
                    if ocr_text.strip():
                        text_parts.append(f"--- Seite {page_num} ---\n{ocr_text}")
                        
                except Exception as e:
                    self.logger.warning(f"Error performing OCR on page {page_num}: {e}")
        
        return "\n\n".join(text_parts)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page number patterns
        if self.config.processing.remove_headers:
            text = re.sub(self.german_patterns['page_numbers'], '', text, flags=re.MULTILINE)
        
        # Remove headers and footers
        if self.config.processing.remove_headers:
            text = re.sub(self.german_patterns['headers'], '', text, flags=re.MULTILINE)
        
        if self.config.processing.remove_footers:
            text = re.sub(self.german_patterns['footers'], '', text, flags=re.MULTILINE)
        
        # Normalize German characters
        text = text.replace('ß', 'ss')  # Optional: normalize sharp s
        
        # Remove special characters but keep German umlauts
        text = re.sub(r'[^\w\säöüÄÖÜß.,!?;:()\[\]{}"\'-]', '', text)
        
        # Normalize whitespace
        if self.config.processing.normalize_whitespace:
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
            text = re.sub(r' +', ' ', text)  # Normalize spaces
        
        return text.strip()
    
    def _create_chunks(self, text: str) -> List[TextChunk]:
        """Create text chunks for processing."""
        chunks = []
        chunk_size = self.config.model.chunk_size
        chunk_overlap = self.config.model.chunk_overlap
        
        # Split text into sentences first
        sentences = re.split(r'[.!?]+', text)
        current_chunk = ""
        chunk_id = 0
        start_char = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                # Create chunk
                chunk = TextChunk(
                    text=current_chunk.strip(),
                    page_number=self._estimate_page_number(start_char, len(text)),
                    chunk_id=f"chunk_{chunk_id:04d}",
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                    language=self.config.processing.default_language
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + " " + sentence
                start_char = start_char + overlap_start
                chunk_id += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk = TextChunk(
                text=current_chunk.strip(),
                page_number=self._estimate_page_number(start_char, len(text)),
                chunk_id=f"chunk_{chunk_id:04d}",
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                language=self.config.processing.default_language
            )
            chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} text chunks")
        return chunks
    
    def _estimate_page_number(self, char_position: int, total_chars: int) -> int:
        """Estimate page number based on character position."""
        # Simple estimation - can be improved with actual page boundaries
        if total_chars == 0:
            return 1
        return max(1, int((char_position / total_chars) * 10) + 1)
    
    def _detect_language(self, pdf_path: Path) -> str:
        """Detect the language of the document."""
        # Simple German detection based on common words
        german_words = ['der', 'die', 'das', 'und', 'in', 'den', 'von', 'zu', 'mit', 'sich']
        
        try:
            # Extract a sample of text for language detection
            with pdfplumber.open(pdf_path) as pdf:
                if pdf.pages:
                    sample_text = pdf.pages[0].extract_text()[:1000].lower()
                    
                    # Count German words
                    german_count = sum(1 for word in german_words if word in sample_text)
                    
                    if german_count > 3:
                        return "de"
                    else:
                        return "en"
        except:
            pass
        
        return self.config.processing.default_language
    
    def _analyze_structure(self, text: str, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Analyze document structure and characteristics."""
        structure = {
            'total_chars': len(text),
            'total_words': len(text.split()),
            'total_sentences': len(re.split(r'[.!?]+', text)),
            'num_chunks': len(chunks),
            'avg_chunk_size': np.mean([len(chunk.text) for chunk in chunks]) if chunks else 0,
            'german_characteristics': {
                'umlaut_count': len(re.findall(self.german_patterns['umlauts'], text)),
                'german_word_count': len(re.findall(self.german_patterns['german_words'], text)),
            },
            'document_sections': self._identify_sections(text)
        }
        
        return structure
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify document sections based on headers and structure."""
        sections = []
        
        # Common German government document section patterns
        section_patterns = [
            r'^(?:\d+\.)?\s*(Einleitung|Introduction)',
            r'^(?:\d+\.)?\s*(Hintergrund|Background)',
            r'^(?:\d+\.)?\s*(Ziele|Objectives)',
            r'^(?:\d+\.)?\s*(Methodik|Methodology)',
            r'^(?:\d+\.)?\s*(Ergebnisse|Results)',
            r'^(?:\d+\.)?\s*(Diskussion|Discussion)',
            r'^(?:\d+\.)?\s*(Schlussfolgerung|Conclusion)',
            r'^(?:\d+\.)?\s*(Anhang|Appendix)',
        ]
        
        lines = text.split('\n')
        current_section = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if line matches any section pattern
            for pattern in section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    if current_section:
                        sections.append(current_section)
                    
                    current_section = {
                        'title': line,
                        'start_line': i,
                        'end_line': i,
                        'content': line
                    }
                    break
            else:
                if current_section:
                    current_section['content'] += '\n' + line
                    current_section['end_line'] = i
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def save_processed_data(self, processed_data: Dict[str, Any], output_path: str):
        """Save processed data to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for easy loading
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        self.logger.info(f"Processed data saved to: {output_path}")
    
    def load_processed_data(self, input_path: str) -> Dict[str, Any]:
        """Load processed data from file."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {input_path}")
        
        import pickle
        with open(input_path, 'rb') as f:
            processed_data = pickle.load(f)
        
        self.logger.info(f"Processed data loaded from: {input_path}")
        return processed_data 