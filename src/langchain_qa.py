"""
LangChain-based Question Answering System.

Implements RAG (Retrieval-Augmented Generation) for answering questions
about German government documents using LangChain framework.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from loguru import logger

from .config import get_config
from .vector_store import VectorStore, SearchResult


@dataclass
class QAAnswer:
    """Represents a question-answer pair with metadata."""
    
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    context_used: List[str]
    processing_time: float
    model_used: str


class LangChainQA:
    """LangChain-based question answering system for German documents."""
    
    def __init__(self, config=None, vector_store=None):
        """Initialize the QA system."""
        self.config = config or get_config()
        self.vector_store = vector_store or VectorStore(config)
        self.logger = logger.bind(name="LangChainQA")
        
        # Initialize language model
        self.llm = self._initialize_llm()
        
        # Initialize QA chain
        self.qa_chain = self._initialize_qa_chain()
        
        # Initialize retriever
        self.retriever = self._initialize_retriever()
        
        self.logger.info("LangChain QA system initialized")
    
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            # Try to use German GPT model
            model_name = self.config.model.german_gpt_model
            
            self.logger.info(f"Loading language model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.config.qa_system.answer_max_length,
                temperature=self.config.model.temperature,
                top_p=self.config.model.top_p,
                top_k=self.config.model.top_k,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            self.logger.info("Language model loaded successfully")
            return llm
            
        except Exception as e:
            self.logger.warning(f"Failed to load German GPT model: {e}")
            self.logger.info("Falling back to default model")
            
            # Fallback to a simpler model or mock
            return self._create_mock_llm()
    
    def _create_mock_llm(self):
        """Create a mock LLM for testing purposes."""
        class MockLLM:
            def __call__(self, prompt):
                return f"Mock response to: {prompt[:100]}..."
        
        return MockLLM()
    
    def _initialize_qa_chain(self):
        """Initialize the QA chain."""
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=self.config.qa_system.qa_prompt_template
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        return qa_chain
    
    def _initialize_retriever(self):
        """Initialize the document retriever."""
        # Create a custom retriever that works with our vector store
        class CustomRetriever:
            def __init__(self, vector_store):
                self.vector_store = vector_store
                self.config = vector_store.config
            
            def get_relevant_documents(self, query: str) -> List[Document]:
                """Get relevant documents for a query."""
                results = self.vector_store.search(query)
                
                documents = []
                for result in results:
                    doc = Document(
                        page_content=result.text,
                        metadata={
                            'chunk_id': result.chunk_id,
                            'page_number': result.page_number,
                            'score': result.score,
                            'start_char': result.start_char,
                            'end_char': result.end_char
                        }
                    )
                    documents.append(doc)
                
                return documents
        
        return CustomRetriever(self.vector_store)
    
    def ask_question(self, question: str, include_sources: bool = None) -> QAAnswer:
        """
        Ask a question and get an answer.
        
        Args:
            question: The question to ask
            include_sources: Whether to include source information
            
        Returns:
            QAAnswer object with answer and metadata
        """
        import time
        start_time = time.time()
        
        self.logger.info(f"Processing question: {question[:50]}...")
        
        # Set default for include_sources
        if include_sources is None:
            include_sources = self.config.qa_system.include_sources
        
        try:
            # Get relevant documents
            relevant_docs = self.retriever.get_relevant_documents(question)
            
            if not relevant_docs:
                return QAAnswer(
                    question=question,
                    answer="Entschuldigung, ich konnte keine relevanten Informationen zu Ihrer Frage finden.",
                    confidence=0.0,
                    sources=[],
                    context_used=[],
                    processing_time=time.time() - start_time,
                    model_used=self.config.model.german_gpt_model
                )
            
            # Prepare context
            context = self._prepare_context(relevant_docs)
            
            # Generate answer
            if hasattr(self.llm, '__call__'):
                # Use the LLM directly
                prompt = self.config.qa_system.qa_prompt_template.format(
                    context=context,
                    question=question
                )
                answer = self.llm(prompt)
            else:
                # Use the QA chain
                result = self.qa_chain({"query": question})
                answer = result.get("result", "Keine Antwort verfügbar.")
            
            # Calculate confidence based on source scores
            confidence = self._calculate_confidence(relevant_docs)
            
            # Prepare sources
            sources = []
            if include_sources:
                sources = self._prepare_sources(relevant_docs)
            
            # Prepare context used
            context_used = [doc.page_content for doc in relevant_docs]
            
            processing_time = time.time() - start_time
            
            return QAAnswer(
                question=question,
                answer=answer,
                confidence=confidence,
                sources=sources,
                context_used=context_used,
                processing_time=processing_time,
                model_used=self.config.model.german_gpt_model
            )
            
        except Exception as e:
            self.logger.error(f"Error processing question: {e}")
            return QAAnswer(
                question=question,
                answer=f"Ein Fehler ist aufgetreten: {str(e)}",
                confidence=0.0,
                sources=[],
                context_used=[],
                processing_time=time.time() - start_time,
                model_used=self.config.model.german_gpt_model
            )
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """Prepare context from relevant documents."""
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Truncate if too long
            content = doc.page_content
            if len(content) > self.config.qa_system.max_context_length // len(documents):
                content = content[:self.config.qa_system.max_context_length // len(documents)] + "..."
            
            context_parts.append(f"Quelle {i+1}:\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _calculate_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score based on source relevance."""
        if not documents:
            return 0.0
        
        # Average the scores from the vector store
        scores = [doc.metadata.get('score', 0.0) for doc in documents]
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range
        confidence = min(1.0, avg_score)
        
        return confidence
    
    def _prepare_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Prepare source information for the answer."""
        sources = []
        
        for i, doc in enumerate(documents):
            source = {
                'source_id': i + 1,
                'chunk_id': doc.metadata.get('chunk_id', f'chunk_{i}'),
                'page_number': doc.metadata.get('page_number', 1),
                'relevance_score': doc.metadata.get('score', 0.0),
                'text_preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source)
        
        return sources
    
    def batch_ask(self, questions: List[str]) -> List[QAAnswer]:
        """
        Ask multiple questions in batch.
        
        Args:
            questions: List of questions to ask
            
        Returns:
            List of QAAnswer objects
        """
        self.logger.info(f"Processing {len(questions)} questions in batch")
        
        answers = []
        for question in questions:
            answer = self.ask_question(question)
            answers.append(answer)
        
        return answers
    
    def ask_with_followup(self, question: str, conversation_history: List[Dict[str, str]] = None) -> QAAnswer:
        """
        Ask a question with conversation history for context.
        
        Args:
            question: The current question
            conversation_history: List of previous Q&A pairs
            
        Returns:
            QAAnswer object
        """
        if not conversation_history:
            return self.ask_question(question)
        
        # Enhance question with context from history
        enhanced_question = self._enhance_question_with_history(question, conversation_history)
        
        return self.ask_question(enhanced_question)
    
    def _enhance_question_with_history(self, question: str, history: List[Dict[str, str]]) -> str:
        """Enhance question with conversation history."""
        # Take the last few exchanges for context
        recent_history = history[-3:]  # Last 3 exchanges
        
        context_parts = []
        for exchange in recent_history:
            if 'question' in exchange and 'answer' in exchange:
                context_parts.append(f"Frage: {exchange['question']}")
                context_parts.append(f"Antwort: {exchange['answer'][:100]}...")
        
        if context_parts:
            context = "\n".join(context_parts)
            enhanced_question = f"Kontext der vorherigen Konversation:\n{context}\n\nAktuelle Frage: {question}"
        else:
            enhanced_question = question
        
        return enhanced_question
    
    def get_answer_with_explanation(self, question: str) -> Dict[str, Any]:
        """
        Get answer with detailed explanation of the reasoning process.
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary with answer, explanation, and metadata
        """
        # Get regular answer
        qa_answer = self.ask_question(question, include_sources=True)
        
        # Generate explanation
        explanation = self._generate_explanation(question, qa_answer)
        
        return {
            'question': question,
            'answer': qa_answer.answer,
            'explanation': explanation,
            'confidence': qa_answer.confidence,
            'sources': qa_answer.sources,
            'processing_time': qa_answer.processing_time,
            'model_used': qa_answer.model_used
        }
    
    def _generate_explanation(self, question: str, qa_answer: QAAnswer) -> str:
        """Generate explanation for the answer."""
        if not qa_answer.sources:
            return "Keine Quellen gefunden, daher kann keine detaillierte Erklärung gegeben werden."
        
        explanation_parts = [
            f"Die Antwort basiert auf {len(qa_answer.sources)} relevanten Dokumentenabschnitten.",
            f"Das Vertrauen in die Antwort beträgt {qa_answer.confidence:.2%}.",
        ]
        
        if qa_answer.sources:
            best_source = max(qa_answer.sources, key=lambda x: x['relevance_score'])
            explanation_parts.append(
                f"Der relevanteste Abschnitt stammt von Seite {best_source['page_number']} "
                f"mit einer Relevanz von {best_source['relevance_score']:.2%}."
            )
        
        return " ".join(explanation_parts)
    
    def validate_answer(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Validate the quality of an answer.
        
        Args:
            question: The original question
            answer: The generated answer
            
        Returns:
            Validation results
        """
        validation = {
            'answer_length': len(answer),
            'has_german_content': bool(re.search(r'[äöüÄÖÜß]', answer)),
            'has_sources': 'Quelle' in answer or 'Seite' in answer,
            'confidence_threshold_met': True,  # Will be updated based on actual confidence
            'quality_score': 0.0
        }
        
        # Calculate quality score
        score = 0.0
        
        # Length score (prefer medium-length answers)
        if 50 <= validation['answer_length'] <= 500:
            score += 0.3
        elif validation['answer_length'] > 0:
            score += 0.1
        
        # German content score
        if validation['has_german_content']:
            score += 0.3
        
        # Source reference score
        if validation['has_sources']:
            score += 0.2
        
        # Completeness score (basic heuristics)
        if any(word in answer.lower() for word in ['ist', 'sind', 'wurde', 'wurden', 'hat', 'haben']):
            score += 0.2
        
        validation['quality_score'] = min(1.0, score)
        
        return validation
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and performance metrics."""
        vector_stats = self.vector_store.get_statistics()
        
        stats = {
            'vector_store': vector_stats,
            'qa_system': {
                'model_name': self.config.model.german_gpt_model,
                'max_answer_length': self.config.qa_system.answer_max_length,
                'confidence_threshold': self.config.qa_system.confidence_threshold,
                'include_sources': self.config.qa_system.include_sources
            },
            'retrieval': {
                'n_results': self.config.vector_store.n_results,
                'score_threshold': self.config.vector_store.score_threshold,
                'similarity_metric': self.config.vector_store.similarity_metric
            }
        }
        
        return stats 