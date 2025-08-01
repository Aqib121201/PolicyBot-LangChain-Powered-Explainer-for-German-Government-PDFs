"""
Unit tests for PDF processor module.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.pdf_processor import PDFProcessor, DocumentMetadata, TextChunk
from src.config import get_config


class TestPDFProcessor:
    """Test cases for PDFProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a PDF processor instance for testing."""
        config = get_config()
        return PDFProcessor(config)
    
    @pytest.fixture
    def sample_text(self):
        """Sample German text for testing."""
        return """
        Bundesregierung Deutschland
        
        Einleitung
        
        Dies ist ein Beispieltext für deutsche Regierungsdokumente.
        Das Dokument enthält wichtige Informationen über die Politik.
        
        Hintergrund
        
        Die deutsche Bundesregierung hat verschiedene Ziele und Maßnahmen.
        Diese werden in diesem Dokument detailliert beschrieben.
        
        Schlussfolgerung
        
        Zusammenfassend kann gesagt werden, dass die Maßnahmen wichtig sind.
        """
    
    def test_initialization(self, processor):
        """Test PDF processor initialization."""
        assert processor is not None
        assert processor.config is not None
        assert hasattr(processor, 'german_patterns')
    
    def test_extract_metadata_success(self, processor):
        """Test successful metadata extraction."""
        with patch('PyPDF2.PdfReader') as mock_reader:
            # Mock PDF reader
            mock_info = {
                '/Title': 'Test Document',
                '/Author': 'Test Author',
                '/Subject': 'Test Subject',
                '/Creator': 'Test Creator',
                '/Producer': 'Test Producer',
                '/CreationDate': '20230101',
                '/ModDate': '20230102'
            }
            mock_reader.return_value.metadata = mock_info
            mock_reader.return_value.pages = [Mock()] * 5
            
            # Mock file path
            mock_path = Mock()
            mock_path.stat.return_value.st_size = 1024
            
            metadata = processor._extract_metadata(mock_path)
            
            assert metadata.title == 'Test Document'
            assert metadata.author == 'Test Author'
            assert metadata.num_pages == 5
            assert metadata.file_size == 1024
    
    def test_extract_metadata_failure(self, processor):
        """Test metadata extraction with failure."""
        with patch('PyPDF2.PdfReader', side_effect=Exception("PDF error")):
            mock_path = Mock()
            mock_path.stat.return_value.st_size = 1024
            mock_path.stem = 'test_document'
            
            metadata = processor._extract_metadata(mock_path)
            
            assert metadata.title == 'test_document'
            assert metadata.author == 'Unknown'
            assert metadata.num_pages == 0
    
    def test_clean_text(self, processor, sample_text):
        """Test text cleaning functionality."""
        cleaned = processor._clean_text(sample_text)
        
        # Check that text is cleaned
        assert 'Bundesregierung' not in cleaned  # Headers removed
        assert len(cleaned) < len(sample_text)  # Some content removed
        assert cleaned.strip()  # No leading/trailing whitespace
    
    def test_create_chunks(self, processor, sample_text):
        """Test text chunking functionality."""
        chunks = processor._create_chunks(sample_text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        
        # Check chunk properties
        for chunk in chunks:
            assert chunk.text
            assert chunk.chunk_id.startswith('chunk_')
            assert chunk.page_number > 0
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
    
    def test_detect_language_german(self, processor):
        """Test German language detection."""
        german_text = "Dies ist ein deutscher Text mit Umlauten äöü und ß."
        
        with patch('pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = german_text
            mock_pdf.return_value.pages = [mock_page]
            
            mock_path = Mock()
            language = processor._detect_language(mock_path)
            
            assert language == "de"
    
    def test_detect_language_english(self, processor):
        """Test English language detection."""
        english_text = "This is an English text without German words."
        
        with patch('pdfplumber.open') as mock_pdf:
            mock_page = Mock()
            mock_page.extract_text.return_value = english_text
            mock_pdf.return_value.pages = [mock_page]
            
            mock_path = Mock()
            language = processor._detect_language(mock_path)
            
            assert language == "en"
    
    def test_analyze_structure(self, processor, sample_text):
        """Test document structure analysis."""
        chunks = processor._create_chunks(sample_text)
        structure = processor._analyze_structure(sample_text, chunks)
        
        assert 'total_chars' in structure
        assert 'total_words' in structure
        assert 'total_sentences' in structure
        assert 'num_chunks' in structure
        assert 'german_characteristics' in structure
        
        assert structure['total_chars'] > 0
        assert structure['total_words'] > 0
        assert structure['total_sentences'] > 0
        assert structure['num_chunks'] == len(chunks)
    
    def test_identify_sections(self, processor, sample_text):
        """Test document section identification."""
        sections = processor._identify_sections(sample_text)
        
        # Should identify some sections
        assert len(sections) > 0
        
        for section in sections:
            assert 'title' in section
            assert 'start_line' in section
            assert 'end_line' in section
            assert 'content' in section
    
    @patch('pdfplumber.open')
    def test_extract_with_pdfplumber(self, mock_pdf, processor):
        """Test PDF text extraction with pdfplumber."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test page content"
        mock_pdf.return_value.pages = [mock_page]
        
        mock_path = Mock()
        text = processor._extract_with_pdfplumber(mock_path)
        
        assert "Test page content" in text
        assert "Seite 1" in text
    
    @patch('PyPDF2.PdfReader')
    def test_extract_with_pypdf2(self, mock_reader, processor):
        """Test PDF text extraction with PyPDF2."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "Test page content"
        mock_reader.return_value.pages = [mock_page]
        
        mock_path = Mock()
        text = processor._extract_with_pypdf2(mock_path)
        
        assert "Test page content" in text
        assert "Seite 1" in text
    
    def test_estimate_page_number(self, processor):
        """Test page number estimation."""
        # Test edge cases
        assert processor._estimate_page_number(0, 1000) == 1
        assert processor._estimate_page_number(500, 1000) > 1
        assert processor._estimate_page_number(1000, 1000) > 1
    
    def test_save_and_load_processed_data(self, processor):
        """Test saving and loading processed data."""
        test_data = {
            'metadata': {'title': 'Test'},
            'chunks': [{'text': 'test chunk'}],
            'structure': {'total_chars': 100}
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Test saving
            processor.save_processed_data(test_data, tmp_path)
            
            # Test loading
            loaded_data = processor.load_processed_data(tmp_path)
            
            assert loaded_data['metadata']['title'] == 'Test'
            assert len(loaded_data['chunks']) == 1
            assert loaded_data['structure']['total_chars'] == 100
            
        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)


class TestDocumentMetadata:
    """Test cases for DocumentMetadata class."""
    
    def test_metadata_creation(self):
        """Test DocumentMetadata creation."""
        metadata = DocumentMetadata(
            title="Test Document",
            author="Test Author",
            subject="Test Subject",
            creator="Test Creator",
            producer="Test Producer",
            creation_date="20230101",
            modification_date="20230102",
            num_pages=10,
            file_size=1024
        )
        
        assert metadata.title == "Test Document"
        assert metadata.author == "Test Author"
        assert metadata.num_pages == 10
        assert metadata.file_size == 1024
        assert metadata.language == "de"  # Default
        assert metadata.document_type == "government"  # Default
        assert metadata.source == "govdata.de"  # Default


class TestTextChunk:
    """Test cases for TextChunk class."""
    
    def test_chunk_creation(self):
        """Test TextChunk creation."""
        chunk = TextChunk(
            text="Test chunk text",
            page_number=1,
            chunk_id="chunk_0001",
            start_char=0,
            end_char=15
        )
        
        assert chunk.text == "Test chunk text"
        assert chunk.page_number == 1
        assert chunk.chunk_id == "chunk_0001"
        assert chunk.start_char == 0
        assert chunk.end_char == 15
        assert chunk.confidence == 1.0  # Default
        assert chunk.language == "de"  # Default


if __name__ == "__main__":
    pytest.main([__file__]) 