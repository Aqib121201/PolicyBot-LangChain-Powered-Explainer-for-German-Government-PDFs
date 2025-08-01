"""
Document Summarizer for German Government Documents.

Provides abstractive and extractive summarization capabilities for
German government documents using transformer models.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger

from .config import get_config


@dataclass
class SummaryResult:
    """Represents a document summary with metadata."""
    
    summary: str
    summary_type: str  # 'abstractive' or 'extractive'
    length: int
    compression_ratio: float
    key_sentences: List[str]
    confidence: float
    processing_time: float
    model_used: str


class DocumentSummarizer:
    """Summarizer for German government documents."""
    
    def __init__(self, config=None):
        """Initialize the summarizer."""
        self.config = config or get_config()
        self.logger = logger.bind(name="DocumentSummarizer")
        
        # Initialize models
        self.summarization_model = self._initialize_summarization_model()
        self.embedding_model = self._initialize_embedding_model()
        
        self.logger.info("Document summarizer initialized")
    
    def _initialize_summarization_model(self):
        """Initialize the summarization model."""
        try:
            model_name = self.config.model.summarization_model
            
            self.logger.info(f"Loading summarization model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Create pipeline
            summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                max_length=self.config.model.max_length,
                min_length=50,
                do_sample=False
            )
            
            self.logger.info("Summarization model loaded successfully")
            return summarizer
            
        except Exception as e:
            self.logger.warning(f"Failed to load summarization model: {e}")
            return None
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model for extractive summarization."""
        try:
            model_name = self.config.model.embedding_model
            
            self.logger.info(f"Loading embedding model: {model_name}")
            
            embedding_model = SentenceTransformer(model_name)
            
            self.logger.info("Embedding model loaded successfully")
            return embedding_model
            
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {e}")
            return None
    
    def summarize_document(self, text: str, summary_type: str = "abstractive", 
                          max_length: int = None, min_length: int = None) -> SummaryResult:
        """
        Summarize a document using the specified method.
        
        Args:
            text: Document text to summarize
            summary_type: 'abstractive' or 'extractive'
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            SummaryResult object
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Generating {summary_type} summary for document")
        
        if not text.strip():
            return SummaryResult(
                summary="Kein Text zum Zusammenfassen verfügbar.",
                summary_type=summary_type,
                length=0,
                compression_ratio=0.0,
                key_sentences=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                model_used=self.config.model.summarization_model
            )
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        if summary_type == "abstractive":
            return self._generate_abstractive_summary(
                cleaned_text, max_length, min_length, start_time
            )
        elif summary_type == "extractive":
            return self._generate_extractive_summary(
                cleaned_text, max_length, start_time
            )
        else:
            raise ValueError(f"Unsupported summary type: {summary_type}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(Bundesregierung|Bundesministerium|Bundestag|Bundesrat)', '', text, flags=re.MULTILINE)
        
        # Normalize German characters
        text = text.replace('ß', 'ss')
        
        # Remove special characters but keep German umlauts
        text = re.sub(r'[^\w\säöüÄÖÜß.,!?;:()\[\]{}"\'-]', '', text)
        
        return text.strip()
    
    def _generate_abstractive_summary(self, text: str, max_length: int = None, 
                                    min_length: int = None, start_time: float = None) -> SummaryResult:
        """Generate abstractive summary using transformer model."""
        if not self.summarization_model:
            return self._fallback_summary(text, "abstractive", start_time)
        
        try:
            # Set default lengths
            max_length = max_length or self.config.model.max_length
            min_length = min_length or 50
            
            # Split text into chunks if too long
            chunks = self._split_text_for_summarization(text, max_length)
            
            summaries = []
            for chunk in chunks:
                if len(chunk.strip()) < 100:  # Skip very short chunks
                    continue
                
                result = self.summarization_model(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                
                if result and len(result) > 0:
                    summaries.append(result[0]['summary_text'])
            
            # Combine summaries if multiple chunks
            if len(summaries) > 1:
                combined_summary = " ".join(summaries)
                # Re-summarize if combined summary is too long
                if len(combined_summary) > max_length * 2:
                    final_result = self.summarization_model(
                        combined_summary,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False
                    )
                    final_summary = final_result[0]['summary_text'] if final_result else combined_summary
                else:
                    final_summary = combined_summary
            else:
                final_summary = summaries[0] if summaries else "Keine Zusammenfassung generiert."
            
            processing_time = time.time() - start_time if start_time else 0.0
            
            return SummaryResult(
                summary=final_summary,
                summary_type="abstractive",
                length=len(final_summary),
                compression_ratio=len(final_summary) / len(text) if text else 0.0,
                key_sentences=self._extract_key_sentences(text),
                confidence=0.8,  # Placeholder confidence
                processing_time=processing_time,
                model_used=self.config.model.summarization_model
            )
            
        except Exception as e:
            self.logger.error(f"Error generating abstractive summary: {e}")
            return self._fallback_summary(text, "abstractive", start_time)
    
    def _generate_extractive_summary(self, text: str, max_length: int = None, 
                                   start_time: float = None) -> SummaryResult:
        """Generate extractive summary using sentence ranking."""
        if not self.embedding_model:
            return self._fallback_summary(text, "extractive", start_time)
        
        try:
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            if len(sentences) < 3:
                return SummaryResult(
                    summary=text,
                    summary_type="extractive",
                    length=len(text),
                    compression_ratio=1.0,
                    key_sentences=sentences,
                    confidence=1.0,
                    processing_time=time.time() - start_time if start_time else 0.0,
                    model_used=self.config.model.embedding_model
                )
            
            # Generate embeddings for sentences
            embeddings = self.embedding_model.encode(sentences)
            
            # Calculate sentence importance scores
            importance_scores = self._calculate_sentence_importance(embeddings, sentences)
            
            # Select top sentences
            max_length = max_length or self.config.model.max_length
            selected_sentences = self._select_top_sentences(
                sentences, importance_scores, max_length
            )
            
            # Combine selected sentences
            summary = " ".join(selected_sentences)
            
            processing_time = time.time() - start_time if start_time else 0.0
            
            return SummaryResult(
                summary=summary,
                summary_type="extractive",
                length=len(summary),
                compression_ratio=len(summary) / len(text) if text else 0.0,
                key_sentences=selected_sentences,
                confidence=0.9,  # High confidence for extractive
                processing_time=processing_time,
                model_used=self.config.model.embedding_model
            )
            
        except Exception as e:
            self.logger.error(f"Error generating extractive summary: {e}")
            return self._fallback_summary(text, "extractive", start_time)
    
    def _split_text_for_summarization(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks suitable for summarization."""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_length:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # German sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _calculate_sentence_importance(self, embeddings: np.ndarray, sentences: List[str]) -> np.ndarray:
        """Calculate importance scores for sentences."""
        # Use TF-IDF inspired scoring
        scores = []
        
        for i, sentence in enumerate(sentences):
            score = 0.0
            
            # Length score (prefer medium-length sentences)
            length = len(sentence.split())
            if 10 <= length <= 30:
                score += 0.3
            elif length > 5:
                score += 0.1
            
            # Position score (prefer sentences from beginning and end)
            position = i / len(sentences)
            if position < 0.2 or position > 0.8:
                score += 0.2
            
            # Keyword score (prefer sentences with important words)
            important_words = ['wichtig', 'zentral', 'haupt', 'grund', 'ziel', 'ergebnis', 'schluss']
            sentence_lower = sentence.lower()
            keyword_count = sum(1 for word in important_words if word in sentence_lower)
            score += keyword_count * 0.1
            
            # German content score
            if re.search(r'[äöüÄÖÜß]', sentence):
                score += 0.1
            
            scores.append(score)
        
        return np.array(scores)
    
    def _select_top_sentences(self, sentences: List[str], scores: np.ndarray, 
                            max_length: int) -> List[str]:
        """Select top sentences based on scores and length constraint."""
        # Sort sentences by score
        sorted_indices = np.argsort(scores)[::-1]
        
        selected_sentences = []
        current_length = 0
        
        for idx in sorted_indices:
            sentence = sentences[idx]
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= max_length:
                selected_sentences.append(sentence)
                current_length += sentence_length
            else:
                break
        
        # Sort by original order
        selected_sentences.sort(key=lambda s: sentences.index(s))
        
        return selected_sentences
    
    def _extract_key_sentences(self, text: str) -> List[str]:
        """Extract key sentences from text."""
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 3:
            return sentences
        
        # Simple heuristic: take first, middle, and last sentences
        key_indices = [0, len(sentences) // 2, len(sentences) - 1]
        key_sentences = [sentences[i] for i in key_indices if i < len(sentences)]
        
        return key_sentences
    
    def _fallback_summary(self, text: str, summary_type: str, start_time: float = None) -> SummaryResult:
        """Generate a fallback summary when models are not available."""
        import time
        
        # Simple extractive summary
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 3:
            summary = text
            key_sentences = sentences
        else:
            # Take first few sentences
            summary = " ".join(sentences[:3])
            key_sentences = sentences[:3]
        
        processing_time = time.time() - start_time if start_time else 0.0
        
        return SummaryResult(
            summary=summary,
            summary_type=summary_type,
            length=len(summary),
            compression_ratio=len(summary) / len(text) if text else 0.0,
            key_sentences=key_sentences,
            confidence=0.5,  # Low confidence for fallback
            processing_time=processing_time,
            model_used="fallback"
        )
    
    def generate_section_summaries(self, text: str, sections: List[Dict[str, Any]]) -> List[SummaryResult]:
        """
        Generate summaries for individual document sections.
        
        Args:
            text: Full document text
            sections: List of section information
            
        Returns:
            List of SummaryResult objects for each section
        """
        section_summaries = []
        
        for section in sections:
            if 'content' in section and section['content']:
                summary = self.summarize_document(
                    section['content'],
                    summary_type="extractive",
                    max_length=200
                )
                section_summaries.append(summary)
        
        return section_summaries
    
    def generate_executive_summary(self, text: str, max_length: int = 300) -> SummaryResult:
        """
        Generate an executive summary suitable for high-level overview.
        
        Args:
            text: Document text
            max_length: Maximum summary length
            
        Returns:
            SummaryResult object
        """
        # First generate a longer summary
        initial_summary = self.summarize_document(
            text,
            summary_type="abstractive",
            max_length=max_length * 2
        )
        
        # Then summarize the summary for executive overview
        executive_summary = self.summarize_document(
            initial_summary.summary,
            summary_type="extractive",
            max_length=max_length
        )
        
        return executive_summary
    
    def compare_summaries(self, text: str) -> Dict[str, Any]:
        """
        Compare different summarization approaches.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with comparison results
        """
        # Generate both types of summaries
        abstractive = self.summarize_document(text, summary_type="abstractive")
        extractive = self.summarize_document(text, summary_type="extractive")
        
        comparison = {
            'abstractive': {
                'summary': abstractive.summary,
                'length': abstractive.length,
                'compression_ratio': abstractive.compression_ratio,
                'confidence': abstractive.confidence,
                'processing_time': abstractive.processing_time
            },
            'extractive': {
                'summary': extractive.summary,
                'length': extractive.length,
                'compression_ratio': extractive.compression_ratio,
                'confidence': extractive.confidence,
                'processing_time': extractive.processing_time
            },
            'comparison': {
                'length_difference': abs(abstractive.length - extractive.length),
                'compression_difference': abs(abstractive.compression_ratio - extractive.compression_ratio),
                'confidence_difference': abs(abstractive.confidence - extractive.confidence),
                'time_difference': abs(abstractive.processing_time - extractive.processing_time)
            }
        }
        
        return comparison
    
    def get_summary_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about the document for summarization.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with statistics
        """
        sentences = self._split_into_sentences(text)
        words = text.split()
        
        stats = {
            'total_sentences': len(sentences),
            'total_words': len(words),
            'total_characters': len(text),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'german_characteristics': {
                'umlaut_count': len(re.findall(r'[äöüÄÖÜß]', text)),
                'german_word_count': len(re.findall(r'\b[A-Za-zäöüÄÖÜß]+\b', text))
            },
            'recommended_summary_length': min(len(text) // 4, 500)  # 25% or 500 chars
        }
        
        return stats 