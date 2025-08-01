"""
Unit tests for document summarizer module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from src.summarizer import DocumentSummarizer, SummaryResult
from src.config import get_config


class TestDocumentSummarizer:
    """Test cases for DocumentSummarizer class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.model.summarization_model = "facebook/bart-large-cnn"
        config.model.embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        config.model.max_length = 512
        config.processing.default_language = "de"
        return config
    
    @pytest.fixture
    def summarizer(self, mock_config):
        """Create a summarizer instance for testing."""
        with patch('src.summarizer.AutoTokenizer'), \
             patch('src.summarizer.AutoModelForSeq2SeqLM'), \
             patch('src.summarizer.pipeline'), \
             patch('src.summarizer.SentenceTransformer'):
            
            summarizer = DocumentSummarizer(mock_config)
            return summarizer
    
    @pytest.fixture
    def sample_text(self):
        """Sample German text for testing."""
        return """
        Die deutsche Bundesregierung hat verschiedene Ziele und Maßnahmen zur Förderung der Wirtschaft entwickelt.
        Diese Politik zielt darauf ab, nachhaltiges Wachstum und soziale Gerechtigkeit zu gewährleisten.
        Deutschland steht vor verschiedenen Herausforderungen im Bereich der Digitalisierung und des Klimawandels.
        Die Regierung hat daher umfassende Maßnahmen beschlossen, um diese Herausforderungen zu bewältigen.
        Die wichtigsten Ziele sind die Förderung der Digitalisierung, Klimaneutralität bis 2045, soziale Gerechtigkeit und wirtschaftliches Wachstum.
        Zur Erreichung dieser Ziele werden Investitionen in digitale Infrastruktur, Förderung erneuerbarer Energien, Ausbau des sozialen Sicherungsnetzes und Unterstützung von Innovationen umgesetzt.
        Die beschlossenen Maßnahmen werden Deutschland dabei helfen, die Herausforderungen der Zukunft zu meistern und eine nachhaltige Entwicklung zu gewährleisten.
        """
    
    def test_initialization(self, summarizer):
        """Test summarizer initialization."""
        assert summarizer is not None
        assert summarizer.config is not None
    
    def test_preprocess_text(self, summarizer, sample_text):
        """Test text preprocessing."""
        cleaned = summarizer._preprocess_text(sample_text)
        
        # Check that text is cleaned
        assert len(cleaned) < len(sample_text)  # Some content removed
        assert cleaned.strip()  # No leading/trailing whitespace
        assert 'Bundesregierung' not in cleaned  # Headers removed
    
    def test_split_into_sentences(self, summarizer, sample_text):
        """Test sentence splitting."""
        sentences = summarizer._split_into_sentences(sample_text)
        
        assert len(sentences) > 0
        assert all(isinstance(s, str) for s in sentences)
        assert all(s.strip() for s in sentences)
    
    def test_split_text_for_summarization(self, summarizer, sample_text):
        """Test text splitting for summarization."""
        chunks = summarizer._split_text_for_summarization(sample_text, 200)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 200 for chunk in chunks)
    
    def test_calculate_sentence_importance(self, summarizer, sample_text):
        """Test sentence importance calculation."""
        sentences = summarizer._split_into_sentences(sample_text)
        
        # Mock embeddings
        mock_embeddings = Mock()
        mock_embeddings.shape = (len(sentences), 384)
        
        with patch.object(summarizer, 'embedding_model') as mock_embedding_model:
            mock_embedding_model.encode.return_value = mock_embeddings
            
            scores = summarizer._calculate_sentence_importance(mock_embeddings, sentences)
            
            assert len(scores) == len(sentences)
            assert all(isinstance(score, (int, float)) for score in scores)
            assert all(score >= 0 for score in scores)
    
    def test_select_top_sentences(self, summarizer, sample_text):
        """Test top sentence selection."""
        sentences = summarizer._split_into_sentences(sample_text)
        scores = [0.5, 0.8, 0.3, 0.9, 0.6]  # Mock scores
        
        selected = summarizer._select_top_sentences(sentences, scores, 200)
        
        assert len(selected) > 0
        assert all(s in sentences for s in selected)
    
    def test_extract_key_sentences(self, summarizer, sample_text):
        """Test key sentence extraction."""
        key_sentences = summarizer._extract_key_sentences(sample_text)
        
        assert len(key_sentences) > 0
        assert all(isinstance(s, str) for s in key_sentences)
    
    def test_fallback_summary(self, summarizer, sample_text):
        """Test fallback summary generation."""
        summary = summarizer._fallback_summary(sample_text, "abstractive", time.time())
        
        assert isinstance(summary, SummaryResult)
        assert summary.summary
        assert summary.summary_type == "abstractive"
        assert summary.confidence == 0.5  # Low confidence for fallback
    
    @patch('src.summarizer.pipeline')
    def test_generate_abstractive_summary_success(self, mock_pipeline, summarizer, sample_text):
        """Test successful abstractive summary generation."""
        # Mock pipeline response
        mock_pipeline.return_value.return_value = [{'summary_text': 'This is a summary.'}]
        
        summary = summarizer._generate_abstractive_summary(sample_text, 200, 50, time.time())
        
        assert isinstance(summary, SummaryResult)
        assert summary.summary == 'This is a summary.'
        assert summary.summary_type == "abstractive"
        assert summary.confidence == 0.8
    
    @patch('src.summarizer.SentenceTransformer')
    def test_generate_extractive_summary_success(self, mock_sentence_transformer, summarizer, sample_text):
        """Test successful extractive summary generation."""
        # Mock sentence transformer
        mock_model = Mock()
        mock_model.encode.return_value = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        summary = summarizer._generate_extractive_summary(sample_text, 200, time.time())
        
        assert isinstance(summary, SummaryResult)
        assert summary.summary
        assert summary.summary_type == "extractive"
        assert summary.confidence == 0.9
    
    def test_summarize_document_abstractive(self, summarizer, sample_text):
        """Test document summarization with abstractive method."""
        with patch.object(summarizer, '_generate_abstractive_summary') as mock_generate:
            mock_generate.return_value = SummaryResult(
                summary="Abstractive summary",
                summary_type="abstractive",
                length=100,
                compression_ratio=0.3,
                key_sentences=["Sentence 1", "Sentence 2"],
                confidence=0.8,
                processing_time=1.0,
                model_used="test-model"
            )
            
            result = summarizer.summarize_document(sample_text, summary_type="abstractive")
            
            assert isinstance(result, SummaryResult)
            assert result.summary == "Abstractive summary"
            assert result.summary_type == "abstractive"
    
    def test_summarize_document_extractive(self, summarizer, sample_text):
        """Test document summarization with extractive method."""
        with patch.object(summarizer, '_generate_extractive_summary') as mock_generate:
            mock_generate.return_value = SummaryResult(
                summary="Extractive summary",
                summary_type="extractive",
                length=150,
                compression_ratio=0.4,
                key_sentences=["Sentence 1", "Sentence 2"],
                confidence=0.9,
                processing_time=0.5,
                model_used="test-model"
            )
            
            result = summarizer.summarize_document(sample_text, summary_type="extractive")
            
            assert isinstance(result, SummaryResult)
            assert result.summary == "Extractive summary"
            assert result.summary_type == "extractive"
    
    def test_summarize_document_invalid_type(self, summarizer, sample_text):
        """Test document summarization with invalid type."""
        with pytest.raises(ValueError, match="Unsupported summary type"):
            summarizer.summarize_document(sample_text, summary_type="invalid")
    
    def test_summarize_document_empty_text(self, summarizer):
        """Test document summarization with empty text."""
        result = summarizer.summarize_document("", summary_type="abstractive")
        
        assert isinstance(result, SummaryResult)
        assert "kein Text" in result.summary
        assert result.confidence == 0.0
    
    def test_generate_section_summaries(self, summarizer, sample_text):
        """Test section summary generation."""
        sections = [
            {'content': 'Section 1 content'},
            {'content': 'Section 2 content'},
            {'content': ''}  # Empty section
        ]
        
        with patch.object(summarizer, 'summarize_document') as mock_summarize:
            mock_summarize.return_value = SummaryResult(
                summary="Section summary",
                summary_type="extractive",
                length=50,
                compression_ratio=0.3,
                key_sentences=["Key sentence"],
                confidence=0.8,
                processing_time=0.5,
                model_used="test-model"
            )
            
            results = summarizer.generate_section_summaries(sample_text, sections)
            
            assert len(results) == 2  # Only non-empty sections
            assert all(isinstance(result, SummaryResult) for result in results)
    
    def test_generate_executive_summary(self, summarizer, sample_text):
        """Test executive summary generation."""
        with patch.object(summarizer, 'summarize_document') as mock_summarize:
            mock_summarize.return_value = SummaryResult(
                summary="Executive summary",
                summary_type="extractive",
                length=100,
                compression_ratio=0.2,
                key_sentences=["Key sentence"],
                confidence=0.9,
                processing_time=1.0,
                model_used="test-model"
            )
            
            result = summarizer.generate_executive_summary(sample_text, max_length=300)
            
            assert isinstance(result, SummaryResult)
            assert result.summary == "Executive summary"
    
    def test_compare_summaries(self, summarizer, sample_text):
        """Test summary comparison."""
        with patch.object(summarizer, 'summarize_document') as mock_summarize:
            # Mock different summary types
            mock_summarize.side_effect = [
                SummaryResult(
                    summary="Abstractive summary",
                    summary_type="abstractive",
                    length=100,
                    compression_ratio=0.3,
                    key_sentences=[],
                    confidence=0.8,
                    processing_time=1.0,
                    model_used="test-model"
                ),
                SummaryResult(
                    summary="Extractive summary",
                    summary_type="extractive",
                    length=150,
                    compression_ratio=0.4,
                    key_sentences=[],
                    confidence=0.9,
                    processing_time=0.5,
                    model_used="test-model"
                )
            ]
            
            comparison = summarizer.compare_summaries(sample_text)
            
            assert 'abstractive' in comparison
            assert 'extractive' in comparison
            assert 'comparison' in comparison
            
            assert comparison['abstractive']['summary'] == "Abstractive summary"
            assert comparison['extractive']['summary'] == "Extractive summary"
    
    def test_get_summary_statistics(self, summarizer, sample_text):
        """Test summary statistics generation."""
        stats = summarizer.get_summary_statistics(sample_text)
        
        assert 'total_sentences' in stats
        assert 'total_words' in stats
        assert 'total_characters' in stats
        assert 'avg_sentence_length' in stats
        assert 'avg_word_length' in stats
        assert 'german_characteristics' in stats
        assert 'recommended_summary_length' in stats
        
        assert stats['total_sentences'] > 0
        assert stats['total_words'] > 0
        assert stats['total_characters'] > 0
        assert stats['recommended_summary_length'] > 0


class TestSummaryResult:
    """Test cases for SummaryResult class."""
    
    def test_summary_result_creation(self):
        """Test SummaryResult creation."""
        result = SummaryResult(
            summary="Test summary",
            summary_type="abstractive",
            length=100,
            compression_ratio=0.3,
            key_sentences=["Sentence 1", "Sentence 2"],
            confidence=0.8,
            processing_time=1.0,
            model_used="test-model"
        )
        
        assert result.summary == "Test summary"
        assert result.summary_type == "abstractive"
        assert result.length == 100
        assert result.compression_ratio == 0.3
        assert len(result.key_sentences) == 2
        assert result.confidence == 0.8
        assert result.processing_time == 1.0
        assert result.model_used == "test-model"


if __name__ == "__main__":
    pytest.main([__file__]) 