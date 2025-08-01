"""
Keyword Extractor and Topic Classification for German Government Documents.

Provides keyword extraction, topic classification, and semantic analysis
for German government documents using NLP techniques.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import spacy
from wordcloud import WordCloud
from loguru import logger

from .config import get_config


@dataclass
class KeywordResult:
    """Represents extracted keywords with metadata."""
    
    keywords: List[Dict[str, Any]]
    keyword_cloud_data: Dict[str, int]
    tfidf_scores: Dict[str, float]
    processing_time: float
    model_used: str


@dataclass
class TopicResult:
    """Represents topic classification results."""
    
    topics: List[Dict[str, Any]]
    dominant_topic: str
    topic_distribution: Dict[str, float]
    confidence: float
    processing_time: float
    model_used: str


@dataclass
class EntityResult:
    """Represents named entity recognition results."""
    
    entities: List[Dict[str, Any]]
    entity_types: Dict[str, List[str]]
    entity_frequencies: Dict[str, int]
    processing_time: float
    model_used: str


class KeywordExtractor:
    """Keyword extraction and topic classification for German documents."""
    
    def __init__(self, config=None):
        """Initialize the keyword extractor."""
        self.config = config or get_config()
        self.logger = logger.bind(name="KeywordExtractor")
        
        # Initialize models
        self.embedding_model = self._initialize_embedding_model()
        self.classification_model = self._initialize_classification_model()
        self.nlp = self._initialize_spacy()
        
        # German government topic categories
        self.topic_categories = {
            'wirtschaft': ['wirtschaft', 'wirtschaftspolitik', 'finanzen', 'steuern', 'handel'],
            'gesundheit': ['gesundheit', 'medizin', 'krankenversicherung', 'arzneimittel'],
            'bildung': ['bildung', 'schule', 'universität', 'ausbildung', 'forschung'],
            'umwelt': ['umwelt', 'klima', 'nachhaltigkeit', 'energie', 'verkehr'],
            'soziales': ['soziales', 'arbeit', 'sozialversicherung', 'rente', 'familie'],
            'sicherheit': ['sicherheit', 'polizei', 'justiz', 'verteidigung', 'innere_sicherheit'],
            'verwaltung': ['verwaltung', 'bürokratie', 'digitalisierung', 'e-government'],
            'international': ['international', 'eu', 'nato', 'aussenpolitik', 'diplomatie'],
            'infrastruktur': ['infrastruktur', 'bau', 'verkehr', 'digital', 'technologie'],
            'demokratie': ['demokratie', 'wahl', 'parlament', 'regierung', 'bürgerrechte']
        }
        
        self.logger.info("Keyword extractor initialized")
    
    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        try:
            model_name = self.config.model.embedding_model
            
            self.logger.info(f"Loading embedding model: {model_name}")
            
            embedding_model = SentenceTransformer(model_name)
            
            self.logger.info("Embedding model loaded successfully")
            return embedding_model
            
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {e}")
            return None
    
    def _initialize_classification_model(self):
        """Initialize the classification model."""
        try:
            model_name = self.config.model.german_bert_model
            
            self.logger.info(f"Loading classification model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(self.topic_categories),
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Create pipeline
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer
            )
            
            self.logger.info("Classification model loaded successfully")
            return classifier
            
        except Exception as e:
            self.logger.warning(f"Failed to load classification model: {e}")
            return None
    
    def _initialize_spacy(self):
        """Initialize spaCy for German language processing."""
        try:
            # Try to load German model
            nlp = spacy.load("de_core_news_sm")
            self.logger.info("spaCy German model loaded successfully")
            return nlp
        except OSError:
            self.logger.warning("German spaCy model not found. Installing...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "de_core_news_sm"])
                nlp = spacy.load("de_core_news_sm")
                self.logger.info("spaCy German model installed and loaded")
                return nlp
            except Exception as e:
                self.logger.warning(f"Failed to install spaCy German model: {e}")
                return None
    
    def extract_keywords(self, text: str, max_keywords: int = None) -> KeywordResult:
        """
        Extract keywords from German text using multiple methods.
        
        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            KeywordResult object
        """
        import time
        start_time = time.time()
        
        self.logger.info("Extracting keywords from text")
        
        max_keywords = max_keywords or self.config.analysis.max_keywords
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Extract keywords using multiple methods
        tfidf_keywords = self._extract_tfidf_keywords(cleaned_text, max_keywords)
        spacy_keywords = self._extract_spacy_keywords(cleaned_text, max_keywords)
        embedding_keywords = self._extract_embedding_keywords(cleaned_text, max_keywords)
        
        # Combine and rank keywords
        combined_keywords = self._combine_keywords(
            tfidf_keywords, spacy_keywords, embedding_keywords
        )
        
        # Create keyword cloud data
        keyword_cloud_data = {kw['keyword']: kw['score'] for kw in combined_keywords}
        
        # Create TF-IDF scores
        tfidf_scores = {kw['keyword']: kw['tfidf_score'] for kw in combined_keywords}
        
        processing_time = time.time() - start_time
        
        return KeywordResult(
            keywords=combined_keywords[:max_keywords],
            keyword_cloud_data=keyword_cloud_data,
            tfidf_scores=tfidf_scores,
            processing_time=processing_time,
            model_used="combined"
        )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for keyword extraction."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep German umlauts
        text = re.sub(r'[^\w\säöüÄÖÜß]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common stop words
        stop_words = {
            'der', 'die', 'das', 'und', 'in', 'den', 'von', 'zu', 'mit', 'sich',
            'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als',
            'auch', 'es', 'an', 'werden', 'aus', 'er', 'hat', 'daß', 'sie',
            'nach', 'wird', 'bei', 'einer', 'um', 'am', 'sind', 'noch', 'wie',
            'einem', 'über', 'einen', 'so', 'zum', 'haben', 'nur', 'oder',
            'aber', 'vor', 'zur', 'bis', 'mehr', 'durch', 'man', 'sein', 'wurde',
            'sehr', 'zum', 'mir', 'bei', 'hatte', 'kann', 'gegen', 'vom',
            'können', 'schon', 'wenn', 'habe', 'ihre', 'dann', 'unter', 'wir',
            'soll', 'ich', 'eines', 'es', 'jahr', 'zwei', 'jahren', 'großen',
            'wieder', 'da', 'mich', 'wurden', 'was', 'mal', 'jetzt', 'auch',
            'nach', 'denn', 'beim', 'seit', 'viele', 'vielen', 'muss', 'war',
            'sind', 'wurde', 'können', 'haben', 'sein', 'werden', 'müssen',
            'soll', 'kann', 'muss', 'wird', 'hat', 'habe', 'hatte', 'wurden'
        }
        
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words and len(word) >= 3]
        
        return ' '.join(filtered_words)
    
    def _extract_tfidf_keywords(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        """Extract keywords using TF-IDF."""
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=max_keywords * 2,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Create keyword list
            keywords = []
            for i, score in enumerate(tfidf_scores):
                if score > 0:
                    keywords.append({
                        'keyword': feature_names[i],
                        'score': float(score),
                        'method': 'tfidf',
                        'tfidf_score': float(score)
                    })
            
            # Sort by score
            keywords.sort(key=lambda x: x['score'], reverse=True)
            
            return keywords[:max_keywords]
            
        except Exception as e:
            self.logger.warning(f"TF-IDF keyword extraction failed: {e}")
            return []
    
    def _extract_spacy_keywords(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        """Extract keywords using spaCy."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            
            keywords = []
            for token in doc:
                # Focus on nouns, proper nouns, and adjectives
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    len(token.text) >= 3):
                    
                    keywords.append({
                        'keyword': token.text,
                        'score': 1.0,  # Placeholder score
                        'method': 'spacy',
                        'pos': token.pos_,
                        'tfidf_score': 0.0
                    })
            
            # Remove duplicates
            seen = set()
            unique_keywords = []
            for kw in keywords:
                if kw['keyword'] not in seen:
                    seen.add(kw['keyword'])
                    unique_keywords.append(kw)
            
            return unique_keywords[:max_keywords]
            
        except Exception as e:
            self.logger.warning(f"spaCy keyword extraction failed: {e}")
            return []
    
    def _extract_embedding_keywords(self, text: str, max_keywords: int) -> List[Dict[str, Any]]:
        """Extract keywords using embedding similarity."""
        if not self.embedding_model:
            return []
        
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return []
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(sentences)
            
            # Calculate sentence similarities
            similarities = []
            for i, emb1 in enumerate(embeddings):
                for j, emb2 in enumerate(embeddings[i+1:], i+1):
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append((similarity, sentences[i], sentences[j]))
            
            # Extract important words from similar sentences
            keywords = []
            for similarity, sent1, sent2 in sorted(similarities, reverse=True)[:max_keywords]:
                # Extract common words
                words1 = set(sent1.split())
                words2 = set(sent2.split())
                common_words = words1.intersection(words2)
                
                for word in common_words:
                    if len(word) >= 3:
                        keywords.append({
                            'keyword': word,
                            'score': float(similarity),
                            'method': 'embedding',
                            'tfidf_score': 0.0
                        })
            
            return keywords[:max_keywords]
            
        except Exception as e:
            self.logger.warning(f"Embedding keyword extraction failed: {e}")
            return []
    
    def _combine_keywords(self, tfidf_keywords: List[Dict], spacy_keywords: List[Dict], 
                         embedding_keywords: List[Dict]) -> List[Dict[str, Any]]:
        """Combine keywords from different methods."""
        # Create keyword dictionary
        keyword_dict = {}
        
        # Add TF-IDF keywords
        for kw in tfidf_keywords:
            keyword = kw['keyword']
            if keyword not in keyword_dict:
                keyword_dict[keyword] = {
                    'keyword': keyword,
                    'score': kw['score'],
                    'methods': [kw['method']],
                    'tfidf_score': kw.get('tfidf_score', 0.0),
                    'spacy_score': 0.0,
                    'embedding_score': 0.0
                }
            else:
                keyword_dict[keyword]['score'] = max(keyword_dict[keyword]['score'], kw['score'])
                keyword_dict[keyword]['methods'].append(kw['method'])
                keyword_dict[keyword]['tfidf_score'] = max(
                    keyword_dict[keyword]['tfidf_score'], kw.get('tfidf_score', 0.0)
                )
        
        # Add spaCy keywords
        for kw in spacy_keywords:
            keyword = kw['keyword']
            if keyword not in keyword_dict:
                keyword_dict[keyword] = {
                    'keyword': keyword,
                    'score': kw['score'],
                    'methods': [kw['method']],
                    'tfidf_score': 0.0,
                    'spacy_score': kw['score'],
                    'embedding_score': 0.0
                }
            else:
                keyword_dict[keyword]['spacy_score'] = kw['score']
                keyword_dict[keyword]['methods'].append(kw['method'])
        
        # Add embedding keywords
        for kw in embedding_keywords:
            keyword = kw['keyword']
            if keyword not in keyword_dict:
                keyword_dict[keyword] = {
                    'keyword': keyword,
                    'score': kw['score'],
                    'methods': [kw['method']],
                    'tfidf_score': 0.0,
                    'spacy_score': 0.0,
                    'embedding_score': kw['score']
                }
            else:
                keyword_dict[keyword]['embedding_score'] = kw['score']
                keyword_dict[keyword]['methods'].append(kw['method'])
        
        # Calculate combined scores
        for keyword, data in keyword_dict.items():
            # Weighted combination
            combined_score = (
                data['tfidf_score'] * 0.4 +
                data['spacy_score'] * 0.3 +
                data['embedding_score'] * 0.3
            )
            data['score'] = combined_score
        
        # Convert to list and sort
        combined_keywords = list(keyword_dict.values())
        combined_keywords.sort(key=lambda x: x['score'], reverse=True)
        
        return combined_keywords
    
    def classify_topics(self, text: str) -> TopicResult:
        """
        Classify topics in German text.
        
        Args:
            text: Text to classify
            
        Returns:
            TopicResult object
        """
        import time
        start_time = time.time()
        
        self.logger.info("Classifying topics in text")
        
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        
        # Extract keywords for topic analysis
        keywords_result = self.extract_keywords(cleaned_text, max_keywords=50)
        
        # Classify using multiple methods
        rule_based_topics = self._rule_based_topic_classification(cleaned_text)
        keyword_based_topics = self._keyword_based_topic_classification(keywords_result.keywords)
        embedding_based_topics = self._embedding_based_topic_classification(cleaned_text)
        
        # Combine topic classifications
        combined_topics = self._combine_topic_classifications(
            rule_based_topics, keyword_based_topics, embedding_based_topics
        )
        
        # Determine dominant topic
        dominant_topic = max(combined_topics, key=lambda x: x['confidence'])
        
        # Create topic distribution
        topic_distribution = {topic['topic']: topic['confidence'] for topic in combined_topics}
        
        processing_time = time.time() - start_time
        
        return TopicResult(
            topics=combined_topics,
            dominant_topic=dominant_topic['topic'],
            topic_distribution=topic_distribution,
            confidence=dominant_topic['confidence'],
            processing_time=processing_time,
            model_used="combined"
        )
    
    def _rule_based_topic_classification(self, text: str) -> List[Dict[str, Any]]:
        """Classify topics using rule-based approach."""
        topics = []
        
        for topic, keywords in self.topic_categories.items():
            score = 0
            for keyword in keywords:
                if keyword in text.lower():
                    score += 1
            
            if score > 0:
                confidence = min(score / len(keywords), 1.0)
                topics.append({
                    'topic': topic,
                    'confidence': confidence,
                    'method': 'rule_based'
                })
        
        return topics
    
    def _keyword_based_topic_classification(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify topics based on extracted keywords."""
        topics = []
        keyword_text = ' '.join([kw['keyword'] for kw in keywords])
        
        for topic, topic_keywords in self.topic_categories.items():
            score = 0
            for keyword in keywords:
                if any(tk in keyword['keyword'] for tk in topic_keywords):
                    score += keyword['score']
            
            if score > 0:
                confidence = min(score / len(topic_keywords), 1.0)
                topics.append({
                    'topic': topic,
                    'confidence': confidence,
                    'method': 'keyword_based'
                })
        
        return topics
    
    def _embedding_based_topic_classification(self, text: str) -> List[Dict[str, Any]]:
        """Classify topics using embedding similarity."""
        if not self.embedding_model:
            return []
        
        try:
            # Create topic embeddings
            topic_embeddings = {}
            for topic, keywords in self.topic_categories.items():
                topic_text = ' '.join(keywords)
                topic_emb = self.embedding_model.encode([topic_text])[0]
                topic_embeddings[topic] = topic_emb
            
            # Encode input text
            text_embedding = self.embedding_model.encode([text])[0]
            
            # Calculate similarities
            topics = []
            for topic, topic_emb in topic_embeddings.items():
                similarity = np.dot(text_embedding, topic_emb) / (
                    np.linalg.norm(text_embedding) * np.linalg.norm(topic_emb)
                )
                
                if similarity > 0.1:  # Threshold
                    topics.append({
                        'topic': topic,
                        'confidence': float(similarity),
                        'method': 'embedding_based'
                    })
            
            return topics
            
        except Exception as e:
            self.logger.warning(f"Embedding-based topic classification failed: {e}")
            return []
    
    def _combine_topic_classifications(self, rule_based: List[Dict], keyword_based: List[Dict], 
                                     embedding_based: List[Dict]) -> List[Dict[str, Any]]:
        """Combine topic classifications from different methods."""
        topic_scores = {}
        
        # Combine scores from all methods
        for topic_data in rule_based + keyword_based + embedding_based:
            topic = topic_data['topic']
            confidence = topic_data['confidence']
            method = topic_data['method']
            
            if topic not in topic_scores:
                topic_scores[topic] = {
                    'topic': topic,
                    'rule_based_score': 0.0,
                    'keyword_based_score': 0.0,
                    'embedding_based_score': 0.0,
                    'count': 0
                }
            
            if method == 'rule_based':
                topic_scores[topic]['rule_based_score'] = confidence
            elif method == 'keyword_based':
                topic_scores[topic]['keyword_based_score'] = confidence
            elif method == 'embedding_based':
                topic_scores[topic]['embedding_based_score'] = confidence
            
            topic_scores[topic]['count'] += 1
        
        # Calculate combined confidence
        combined_topics = []
        for topic, scores in topic_scores.items():
            combined_confidence = (
                scores['rule_based_score'] * 0.3 +
                scores['keyword_based_score'] * 0.4 +
                scores['embedding_based_score'] * 0.3
            )
            
            combined_topics.append({
                'topic': topic,
                'confidence': combined_confidence,
                'method': 'combined',
                'rule_based_score': scores['rule_based_score'],
                'keyword_based_score': scores['keyword_based_score'],
                'embedding_based_score': scores['embedding_based_score']
            })
        
        # Sort by confidence
        combined_topics.sort(key=lambda x: x['confidence'], reverse=True)
        
        return combined_topics
    
    def extract_entities(self, text: str) -> EntityResult:
        """
        Extract named entities from German text.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            EntityResult object
        """
        import time
        start_time = time.time()
        
        self.logger.info("Extracting named entities from text")
        
        if not self.nlp:
            return EntityResult(
                entities=[],
                entity_types={},
                entity_frequencies={},
                processing_time=time.time() - start_time,
                model_used="none"
            )
        
        try:
            doc = self.nlp(text)
            
            entities = []
            entity_types = {}
            entity_frequencies = Counter()
            
            for ent in doc.ents:
                entity_data = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_)
                }
                entities.append(entity_data)
                
                # Group by entity type
                if ent.label_ not in entity_types:
                    entity_types[ent.label_] = []
                entity_types[ent.label_].append(ent.text)
                
                # Count frequencies
                entity_frequencies[ent.text] += 1
            
            processing_time = time.time() - start_time
            
            return EntityResult(
                entities=entities,
                entity_types=entity_types,
                entity_frequencies=dict(entity_frequencies),
                processing_time=processing_time,
                model_used="spacy"
            )
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return EntityResult(
                entities=[],
                entity_types={},
                entity_frequencies={},
                processing_time=time.time() - start_time,
                model_used="error"
            )
    
    def generate_keyword_cloud(self, keyword_data: Dict[str, int], 
                             output_path: str = None) -> str:
        """
        Generate a keyword cloud visualization.
        
        Args:
            keyword_data: Dictionary of keywords and their scores
            output_path: Path to save the image
            
        Returns:
            Path to the generated image
        """
        try:
            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis',
                font_path=None  # Use default font
            ).generate_from_frequencies(keyword_data)
            
            # Save or return
            if output_path:
                wordcloud.to_file(output_path)
                self.logger.info(f"Keyword cloud saved to: {output_path}")
                return output_path
            else:
                # Return as base64 string for web display
                import io
                import base64
                
                img_buffer = io.BytesIO()
                wordcloud.to_image().save(img_buffer, format='PNG')
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                
                return f"data:image/png;base64,{img_str}"
                
        except Exception as e:
            self.logger.error(f"Keyword cloud generation failed: {e}")
            return ""
    
    def get_analysis_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get comprehensive analysis statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with analysis statistics
        """
        # Extract keywords
        keywords_result = self.extract_keywords(text)
        
        # Classify topics
        topics_result = self.classify_topics(text)
        
        # Extract entities
        entities_result = self.extract_entities(text)
        
        # Calculate statistics
        stats = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(re.split(r'[.!?]+', text)),
            'keywords': {
                'total_keywords': len(keywords_result.keywords),
                'avg_keyword_score': np.mean([kw['score'] for kw in keywords_result.keywords]) if keywords_result.keywords else 0,
                'top_keywords': [kw['keyword'] for kw in keywords_result.keywords[:10]]
            },
            'topics': {
                'dominant_topic': topics_result.dominant_topic,
                'topic_confidence': topics_result.confidence,
                'topic_distribution': topics_result.topic_distribution
            },
            'entities': {
                'total_entities': len(entities_result.entities),
                'entity_types': list(entities_result.entity_types.keys()),
                'most_frequent_entities': sorted(
                    entities_result.entity_frequencies.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
            },
            'processing_times': {
                'keywords': keywords_result.processing_time,
                'topics': topics_result.processing_time,
                'entities': entities_result.processing_time
            }
        }
        
        return stats 