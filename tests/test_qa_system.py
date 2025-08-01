"""
Unit tests for Q&A system module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import time

from src.langchain_qa import LangChainQA, QAAnswer
from src.vector_store import VectorStore, SearchResult
from src.config import get_config


class TestLangChainQA:
    """Test cases for LangChainQA class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock()
        config.model.german_gpt_model = "microsoft/DialoGPT-medium-german"
        config.qa_system.answer_max_length = 500
        config.qa_system.qa_prompt_template = "Context: {context}\nQuestion: {question}\nAnswer:"
        config.qa_system.include_sources = True
        config.qa_system.confidence_threshold = 0.8
        config.vector_store.n_results = 5
        config.vector_store.score_threshold = 0.7
        return config
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        vector_store = Mock(spec=VectorStore)
        vector_store.config = Mock()
        return vector_store
    
    @pytest.fixture
    def qa_system(self, mock_config, mock_vector_store):
        """Create a Q&A system instance for testing."""
        with patch('src.langchain_qa.AutoTokenizer'), \
             patch('src.langchain_qa.AutoModelForCausalLM'), \
             patch('src.langchain_qa.pipeline'), \
             patch('src.langchain_qa.HuggingFacePipeline'):
            
            qa = LangChainQA(mock_config, mock_vector_store)
            return qa
    
    def test_initialization(self, qa_system):
        """Test Q&A system initialization."""
        assert qa_system is not None
        assert qa_system.config is not None
        assert qa_system.vector_store is not None
    
    def test_ask_question_success(self, qa_system):
        """Test successful question asking."""
        # Mock search results
        mock_search_results = [
            SearchResult(
                chunk_id="chunk_001",
                text="This is relevant context about the topic.",
                score=0.9,
                page_number=1,
                start_char=0,
                end_char=50
            )
        ]
        
        qa_system.vector_store.search.return_value = mock_search_results
        
        # Mock LLM response
        qa_system.llm = Mock()
        qa_system.llm.__call__ = Mock(return_value="This is the answer to your question.")
        
        # Test question asking
        question = "What is the main topic?"
        answer = qa_system.ask_question(question)
        
        assert isinstance(answer, QAAnswer)
        assert answer.question == question
        assert "answer" in answer.answer.lower()
        assert answer.confidence > 0
        assert answer.processing_time > 0
        assert len(answer.sources) > 0
    
    def test_ask_question_no_results(self, qa_system):
        """Test question asking with no search results."""
        qa_system.vector_store.search.return_value = []
        
        question = "What is the main topic?"
        answer = qa_system.ask_question(question)
        
        assert isinstance(answer, QAAnswer)
        assert "keine relevanten Informationen" in answer.answer
        assert answer.confidence == 0.0
        assert len(answer.sources) == 0
    
    def test_ask_question_with_error(self, qa_system):
        """Test question asking with error."""
        qa_system.vector_store.search.side_effect = Exception("Search error")
        
        question = "What is the main topic?"
        answer = qa_system.ask_question(question)
        
        assert isinstance(answer, QAAnswer)
        assert "Fehler" in answer.answer
        assert answer.confidence == 0.0
    
    def test_batch_ask(self, qa_system):
        """Test batch question asking."""
        # Mock search results
        mock_search_results = [
            SearchResult(
                chunk_id="chunk_001",
                text="This is relevant context.",
                score=0.9,
                page_number=1,
                start_char=0,
                end_char=50
            )
        ]
        
        qa_system.vector_store.search.return_value = mock_search_results
        
        # Mock LLM response
        qa_system.llm = Mock()
        qa_system.llm.__call__ = Mock(return_value="This is the answer.")
        
        # Test batch asking
        questions = ["Question 1?", "Question 2?"]
        answers = qa_system.batch_ask(questions)
        
        assert len(answers) == 2
        assert all(isinstance(answer, QAAnswer) for answer in answers)
        assert answers[0].question == "Question 1?"
        assert answers[1].question == "Question 2?"
    
    def test_ask_with_followup(self, qa_system):
        """Test question asking with conversation history."""
        # Mock search results
        mock_search_results = [
            SearchResult(
                chunk_id="chunk_001",
                text="This is relevant context.",
                score=0.9,
                page_number=1,
                start_char=0,
                end_char=50
            )
        ]
        
        qa_system.vector_store.search.return_value = mock_search_results
        
        # Mock LLM response
        qa_system.llm = Mock()
        qa_system.llm.__call__ = Mock(return_value="This is the answer.")
        
        # Test with history
        question = "What about this?"
        history = [
            {'question': 'Previous question?', 'answer': 'Previous answer.'}
        ]
        
        answer = qa_system.ask_with_followup(question, history)
        
        assert isinstance(answer, QAAnswer)
        assert "Kontext" in qa_system.llm.__call__.call_args[0][0]  # Enhanced question
    
    def test_get_answer_with_explanation(self, qa_system):
        """Test getting answer with explanation."""
        # Mock search results
        mock_search_results = [
            SearchResult(
                chunk_id="chunk_001",
                text="This is relevant context.",
                score=0.9,
                page_number=1,
                start_char=0,
                end_char=50
            )
        ]
        
        qa_system.vector_store.search.return_value = mock_search_results
        
        # Mock LLM response
        qa_system.llm = Mock()
        qa_system.llm.__call__ = Mock(return_value="This is the answer.")
        
        # Test with explanation
        question = "What is the main topic?"
        result = qa_system.get_answer_with_explanation(question)
        
        assert 'question' in result
        assert 'answer' in result
        assert 'explanation' in result
        assert 'confidence' in result
        assert 'sources' in result
    
    def test_validate_answer(self, qa_system):
        """Test answer validation."""
        question = "What is the main topic?"
        answer = "Dies ist eine deutsche Antwort mit Umlauten äöü."
        
        validation = qa_system.validate_answer(question, answer)
        
        assert 'answer_length' in validation
        assert 'has_german_content' in validation
        assert 'has_sources' in validation
        assert 'quality_score' in validation
        
        assert validation['has_german_content'] == True
        assert validation['quality_score'] > 0
    
    def test_get_system_stats(self, qa_system):
        """Test getting system statistics."""
        # Mock vector store stats
        qa_system.vector_store.get_statistics.return_value = {
            'total_chunks': 100,
            'index_size': 100,
            'dimension': 384
        }
        
        stats = qa_system.get_system_stats()
        
        assert 'vector_store' in stats
        assert 'qa_system' in stats
        assert 'retrieval' in stats
        
        assert stats['vector_store']['total_chunks'] == 100
        assert stats['qa_system']['model_name'] == "microsoft/DialoGPT-medium-german"
    
    def test_prepare_context(self, qa_system):
        """Test context preparation."""
        from langchain.schema import Document
        
        documents = [
            Document(
                page_content="First document content.",
                metadata={'score': 0.9}
            ),
            Document(
                page_content="Second document content.",
                metadata={'score': 0.8}
            )
        ]
        
        context = qa_system._prepare_context(documents)
        
        assert "Quelle 1" in context
        assert "Quelle 2" in context
        assert "First document content" in context
        assert "Second document content" in context
    
    def test_calculate_confidence(self, qa_system):
        """Test confidence calculation."""
        from langchain.schema import Document
        
        documents = [
            Document(metadata={'score': 0.9}),
            Document(metadata={'score': 0.8}),
            Document(metadata={'score': 0.7})
        ]
        
        confidence = qa_system._calculate_confidence(documents)
        
        assert 0 <= confidence <= 1
        assert confidence == 0.8  # Average of scores
    
    def test_prepare_sources(self, qa_system):
        """Test source preparation."""
        from langchain.schema import Document
        
        documents = [
            Document(
                page_content="This is the first source content.",
                metadata={
                    'chunk_id': 'chunk_001',
                    'page_number': 1,
                    'score': 0.9
                }
            ),
            Document(
                page_content="This is the second source content.",
                metadata={
                    'chunk_id': 'chunk_002',
                    'page_number': 2,
                    'score': 0.8
                }
            )
        ]
        
        sources = qa_system._prepare_sources(documents)
        
        assert len(sources) == 2
        assert sources[0]['source_id'] == 1
        assert sources[0]['chunk_id'] == 'chunk_001'
        assert sources[0]['page_number'] == 1
        assert sources[0]['relevance_score'] == 0.9
        assert "first source content" in sources[0]['text_preview']
    
    def test_enhance_question_with_history(self, qa_system):
        """Test question enhancement with history."""
        question = "What about this?"
        history = [
            {'question': 'Previous question?', 'answer': 'Previous answer.'},
            {'question': 'Another question?', 'answer': 'Another answer.'}
        ]
        
        enhanced = qa_system._enhance_question_with_history(question, history)
        
        assert "Kontext" in enhanced
        assert "Previous question" in enhanced
        assert "Previous answer" in enhanced
        assert "Aktuelle Frage" in enhanced
        assert question in enhanced
    
    def test_generate_explanation(self, qa_system):
        """Test explanation generation."""
        # Create a mock QA answer
        answer = QAAnswer(
            question="What is the topic?",
            answer="This is the answer.",
            confidence=0.85,
            sources=[
                {
                    'source_id': 1,
                    'page_number': 1,
                    'relevance_score': 0.9
                }
            ],
            context_used=["Context 1", "Context 2"],
            processing_time=1.5,
            model_used="test-model"
        )
        
        explanation = qa_system._generate_explanation("What is the topic?", answer)
        
        assert "relevanten Dokumentenabschnitten" in explanation
        assert "85%" in explanation  # Confidence percentage
        assert "Seite 1" in explanation
        assert "90%" in explanation  # Relevance percentage


class TestQAAnswer:
    """Test cases for QAAnswer class."""
    
    def test_qa_answer_creation(self):
        """Test QAAnswer creation."""
        answer = QAAnswer(
            question="What is the topic?",
            answer="This is the answer.",
            confidence=0.85,
            sources=[{'source_id': 1}],
            context_used=["Context 1"],
            processing_time=1.5,
            model_used="test-model"
        )
        
        assert answer.question == "What is the topic?"
        assert answer.answer == "This is the answer."
        assert answer.confidence == 0.85
        assert len(answer.sources) == 1
        assert len(answer.context_used) == 1
        assert answer.processing_time == 1.5
        assert answer.model_used == "test-model"


if __name__ == "__main__":
    pytest.main([__file__]) 