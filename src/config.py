"""
Configuration management for PolicyBot.

Centralized configuration for all system parameters, model paths, and settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for language models and embeddings."""
    
    # German language models
    german_bert_model: str = "bert-base-german-cased"
    german_gpt_model: str = "microsoft/DialoGPT-medium-german"
    
    # Multilingual models
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    summarization_model: str = "facebook/bart-large-cnn"
    
    # Model parameters
    max_length: int = 512
    chunk_size: int = 1000
    chunk_overlap: int = 200
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9


@dataclass
class VectorStoreConfig:
    """Configuration for vector database operations."""
    
    index_type: str = "faiss"
    similarity_metric: str = "cosine"
    n_results: int = 5
    score_threshold: float = 0.7
    dimension: int = 384  # For multilingual-MiniLM-L12-v2


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    
    # PDF processing
    ocr_enabled: bool = True
    ocr_language: str = "deu"
    min_confidence: float = 0.6
    
    # Text cleaning
    remove_headers: bool = True
    remove_footers: bool = True
    normalize_whitespace: bool = True
    
    # Chunking
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    
    # Language detection
    default_language: str = "de"
    supported_languages: list = field(default_factory=lambda: ["de", "en"])


@dataclass
class QASystemConfig:
    """Configuration for question-answering system."""
    
    # RAG parameters
    retrieval_strategy: str = "similarity_search"
    rerank_results: bool = True
    max_context_length: int = 4000
    
    # Answer generation
    answer_max_length: int = 500
    include_sources: bool = True
    confidence_threshold: float = 0.8
    
    # Prompt templates
    qa_prompt_template: str = """
    Basierend auf dem folgenden Kontext, beantworte die Frage auf Deutsch.
    
    Kontext: {context}
    
    Frage: {question}
    
    Antwort:"""
    
    summarization_prompt_template: str = """
    Fasse den folgenden deutschen Text zusammen:
    
    {text}
    
    Zusammenfassung:"""


@dataclass
class AnalysisConfig:
    """Configuration for document analysis features."""
    
    # Keyword extraction
    keyword_extraction_enabled: bool = True
    max_keywords: int = 20
    min_keyword_length: int = 3
    
    # Topic classification
    topic_classification_enabled: bool = True
    num_topics: int = 10
    
    # Sentiment analysis
    sentiment_analysis_enabled: bool = True
    
    # Named entity recognition
    ner_enabled: bool = True
    entity_types: list = field(default_factory=lambda: ["PERSON", "ORG", "LOC", "DATE"])


@dataclass
class PathsConfig:
    """Configuration for file and directory paths."""
    
    # Base directories
    base_dir: Path = Path(__file__).parent.parent
    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    visualizations_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    
    # Data subdirectories
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    external_data_dir: Path = field(init=False)
    
    # Model subdirectories
    embeddings_dir: Path = field(init=False)
    classifiers_dir: Path = field(init=False)
    summarizers_dir: Path = field(init=False)
    
    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.visualizations_dir = self.base_dir / "visualizations"
        self.logs_dir = self.base_dir / "logs"
        
        # Data subdirectories
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.external_data_dir = self.data_dir / "external"
        
        # Model subdirectories
        self.embeddings_dir = self.models_dir / "embeddings"
        self.classifiers_dir = self.models_dir / "classifiers"
        self.summarizers_dir = self.models_dir / "summarizers"
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            self.data_dir, self.models_dir, self.visualizations_dir, self.logs_dir,
            self.raw_data_dir, self.processed_data_dir, self.external_data_dir,
            self.embeddings_dir, self.classifiers_dir, self.summarizers_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[Path] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class APIConfig:
    """Configuration for external APIs."""
    
    # OpenAI (if using)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # HuggingFace
    huggingface_token: Optional[str] = None
    
    # GovData API
    govdata_base_url: str = "https://www.govdata.de"
    govdata_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment variables."""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
        self.govdata_api_key = os.getenv("GOVDATA_API_KEY")


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults."""
        self.model = ModelConfig()
        self.vector_store = VectorStoreConfig()
        self.processing = ProcessingConfig()
        self.qa_system = QASystemConfig()
        self.analysis = AnalysisConfig()
        self.paths = PathsConfig()
        self.logging = LoggingConfig()
        self.api = APIConfig()
        
        # Load from file if provided
        if config_path:
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                self._update_from_dict(config_data)
    
    def save_to_file(self, config_path: str):
        """Save current configuration to YAML file."""
        config_data = self.to_dict()
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        for section_name, section_data in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'vector_store': self.vector_store.__dict__,
            'processing': self.processing.__dict__,
            'qa_system': self.qa_system.__dict__,
            'analysis': self.analysis.__dict__,
            'logging': self.logging.__dict__,
            'api': {k: v for k, v in self.api.__dict__.items() 
                   if not k.endswith('_key')}  # Don't save API keys
        }
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        # Check if required directories exist
        if not self.paths.base_dir.exists():
            raise ValueError(f"Base directory does not exist: {self.paths.base_dir}")
        
        # Check model parameters
        if self.model.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        if self.model.chunk_overlap >= self.model.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        # Check vector store parameters
        if self.vector_store.n_results <= 0:
            raise ValueError("Number of results must be positive")
        
        return True


# Global configuration instance
config = Config()

# Convenience functions
def get_config() -> Config:
    """Get the global configuration instance."""
    return config

def update_config(config_dict: Dict[str, Any]):
    """Update global configuration from dictionary."""
    config._update_from_dict(config_dict)

def save_config(config_path: str):
    """Save global configuration to file."""
    config.save_to_file(config_path)

def load_config(config_path: str):
    """Load global configuration from file."""
    config.load_from_file(config_path) 