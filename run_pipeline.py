#!/usr/bin/env python3
"""
PolicyBot Pipeline Orchestrator.

Main entry point for processing German government PDFs with comprehensive
analysis including Q&A, summarization, and keyword extraction.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time

from loguru import logger

from src.config import get_config, save_config
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.langchain_qa import LangChainQA
from src.summarizer import DocumentSummarizer
from src.keyword_extractor import KeywordExtractor


class PolicyBotPipeline:
    """Main pipeline for processing German government documents."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the pipeline."""
        self.config = get_config()
        if config_path:
            self.config.load_from_file(config_path)
        
        # Initialize components
        self.pdf_processor = PDFProcessor(self.config)
        self.vector_store = VectorStore(self.config)
        self.qa_system = LangChainQA(self.config, self.vector_store)
        self.summarizer = DocumentSummarizer(self.config)
        self.keyword_extractor = KeywordExtractor(self.config)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("PolicyBot Pipeline initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level=self.config.logging.level
        )
        
        # Add file handler
        if self.config.logging.file_path:
            logger.add(
                self.config.logging.file_path,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
                level=self.config.logging.level,
                rotation=self.config.logging.max_file_size,
                retention=self.config.logging.backup_count
            )
    
    def process_document(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single PDF document through the complete pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save results (optional)
            
        Returns:
            Dictionary containing all processing results
        """
        start_time = time.time()
        
        logger.info(f"Starting document processing: {pdf_path}")
        
        try:
            # Step 1: PDF Processing
            logger.info("Step 1: Processing PDF")
            pdf_results = self.pdf_processor.process_pdf(pdf_path)
            
            # Step 2: Vector Store Indexing
            logger.info("Step 2: Indexing document in vector store")
            self.vector_store.add_documents(pdf_results['chunks'])
            
            # Step 3: Document Summarization
            logger.info("Step 3: Generating document summaries")
            summary_results = self._generate_summaries(pdf_results['cleaned_text'])
            
            # Step 4: Keyword Extraction and Topic Classification
            logger.info("Step 4: Extracting keywords and classifying topics")
            analysis_results = self._perform_analysis(pdf_results['cleaned_text'])
            
            # Step 5: Sample Q&A
            logger.info("Step 5: Testing Q&A system")
            qa_results = self._test_qa_system(pdf_results['cleaned_text'])
            
            # Compile results
            results = {
                'document_info': {
                    'file_path': pdf_path,
                    'metadata': pdf_results['metadata'].__dict__,
                    'structure': pdf_results['structure'],
                    'processing_time': time.time() - start_time
                },
                'summarization': summary_results,
                'analysis': analysis_results,
                'qa_samples': qa_results,
                'vector_store_stats': self.vector_store.get_statistics(),
                'system_stats': self.qa_system.get_system_stats()
            }
            
            # Save results if output directory specified
            if output_dir:
                self._save_results(results, output_dir, Path(pdf_path).stem)
            
            logger.info(f"Document processing completed in {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def _generate_summaries(self, text: str) -> Dict[str, Any]:
        """Generate different types of summaries."""
        results = {}
        
        # Abstractive summary
        try:
            abstractive = self.summarizer.summarize_document(text, summary_type="abstractive")
            results['abstractive'] = {
                'summary': abstractive.summary,
                'length': abstractive.length,
                'compression_ratio': abstractive.compression_ratio,
                'confidence': abstractive.confidence,
                'processing_time': abstractive.processing_time
            }
        except Exception as e:
            logger.warning(f"Abstractive summarization failed: {e}")
            results['abstractive'] = {'error': str(e)}
        
        # Extractive summary
        try:
            extractive = self.summarizer.summarize_document(text, summary_type="extractive")
            results['extractive'] = {
                'summary': extractive.summary,
                'length': extractive.length,
                'compression_ratio': extractive.compression_ratio,
                'confidence': extractive.confidence,
                'processing_time': extractive.processing_time
            }
        except Exception as e:
            logger.warning(f"Extractive summarization failed: {e}")
            results['extractive'] = {'error': str(e)}
        
        # Executive summary
        try:
            executive = self.summarizer.generate_executive_summary(text)
            results['executive'] = {
                'summary': executive.summary,
                'length': executive.length,
                'compression_ratio': executive.compression_ratio,
                'confidence': executive.confidence,
                'processing_time': executive.processing_time
            }
        except Exception as e:
            logger.warning(f"Executive summarization failed: {e}")
            results['executive'] = {'error': str(e)}
        
        return results
    
    def _perform_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive document analysis."""
        results = {}
        
        # Keyword extraction
        try:
            keywords = self.keyword_extractor.extract_keywords(text)
            results['keywords'] = {
                'keywords': keywords.keywords[:10],  # Top 10 keywords
                'total_keywords': len(keywords.keywords),
                'processing_time': keywords.processing_time
            }
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            results['keywords'] = {'error': str(e)}
        
        # Topic classification
        try:
            topics = self.keyword_extractor.classify_topics(text)
            results['topics'] = {
                'dominant_topic': topics.dominant_topic,
                'topic_distribution': topics.topic_distribution,
                'confidence': topics.confidence,
                'processing_time': topics.processing_time
            }
        except Exception as e:
            logger.warning(f"Topic classification failed: {e}")
            results['topics'] = {'error': str(e)}
        
        # Entity extraction
        try:
            entities = self.keyword_extractor.extract_entities(text)
            results['entities'] = {
                'total_entities': len(entities.entities),
                'entity_types': list(entities.entity_types.keys()),
                'most_frequent': sorted(
                    entities.entity_frequencies.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10],
                'processing_time': entities.processing_time
            }
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            results['entities'] = {'error': str(e)}
        
        # Overall statistics
        try:
            stats = self.keyword_extractor.get_analysis_statistics(text)
            results['statistics'] = stats
        except Exception as e:
            logger.warning(f"Statistics calculation failed: {e}")
            results['statistics'] = {'error': str(e)}
        
        return results
    
    def _test_qa_system(self, text: str) -> Dict[str, Any]:
        """Test the Q&A system with sample questions."""
        sample_questions = [
            "Was ist das Hauptthema dieses Dokuments?",
            "Welche Ziele werden in diesem Dokument genannt?",
            "Welche Maßnahmen werden vorgeschlagen?",
            "Wer ist für die Umsetzung verantwortlich?",
            "Welche Zeitpläne werden erwähnt?"
        ]
        
        results = {}
        
        for question in sample_questions:
            try:
                answer = self.qa_system.ask_question(question)
                results[question] = {
                    'answer': answer.answer,
                    'confidence': answer.confidence,
                    'processing_time': answer.processing_time,
                    'sources_count': len(answer.sources)
                }
            except Exception as e:
                logger.warning(f"Q&A failed for question '{question}': {e}")
                results[question] = {'error': str(e)}
        
        return results
    
    def _save_results(self, results: Dict[str, Any], output_dir: str, document_name: str):
        """Save processing results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_path / f"{document_name}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save vector store
        vector_store_path = output_path / f"{document_name}_vector_store"
        self.vector_store.save(str(vector_store_path))
        
        # Generate keyword cloud if available
        if 'analysis' in results and 'keywords' in results['analysis']:
            try:
                keyword_data = results['analysis']['keywords'].get('keywords', [])
                if keyword_data:
                    cloud_data = {kw['keyword']: kw['score'] for kw in keyword_data}
                    cloud_path = output_path / f"{document_name}_keyword_cloud.png"
                    self.keyword_extractor.generate_keyword_cloud(cloud_data, str(cloud_path))
            except Exception as e:
                logger.warning(f"Keyword cloud generation failed: {e}")
        
        logger.info(f"Results saved to: {output_path}")
    
    def batch_process(self, pdf_directory: str, output_dir: str) -> Dict[str, Any]:
        """
        Process multiple PDF documents in batch.
        
        Args:
            pdf_directory: Directory containing PDF files
            output_dir: Directory to save results
            
        Returns:
            Dictionary with batch processing results
        """
        pdf_dir = Path(pdf_directory)
        output_path = Path(output_dir)
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        batch_results = {
            'total_files': len(pdf_files),
            'processed_files': 0,
            'failed_files': 0,
            'results': {},
            'summary_stats': {}
        }
        
        start_time = time.time()
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing: {pdf_file.name}")
                
                # Process individual document
                result = self.process_document(
                    str(pdf_file), 
                    str(output_path / pdf_file.stem)
                )
                
                batch_results['results'][pdf_file.name] = result
                batch_results['processed_files'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                batch_results['failed_files'] += 1
                batch_results['results'][pdf_file.name] = {'error': str(e)}
        
        # Calculate summary statistics
        batch_results['summary_stats'] = self._calculate_batch_stats(batch_results['results'])
        batch_results['total_processing_time'] = time.time() - start_time
        
        # Save batch results
        batch_json_path = output_path / "batch_results.json"
        with open(batch_json_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Batch processing completed: {batch_results['processed_files']} successful, "
                   f"{batch_results['failed_files']} failed")
        
        return batch_results
    
    def _calculate_batch_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for batch processing results."""
        successful_results = [r for r in results.values() if 'error' not in r]
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
        
        stats = {
            'total_documents': len(successful_results),
            'avg_processing_time': 0,
            'topic_distribution': {},
            'avg_document_length': 0,
            'common_keywords': [],
            'avg_confidence_scores': {}
        }
        
        # Calculate averages
        total_time = sum(r['document_info']['processing_time'] for r in successful_results)
        stats['avg_processing_time'] = total_time / len(successful_results)
        
        # Document lengths
        lengths = [r['document_info']['structure']['total_chars'] for r in successful_results]
        stats['avg_document_length'] = sum(lengths) / len(lengths)
        
        # Topic distribution
        topic_counts = {}
        for r in successful_results:
            if 'analysis' in r and 'topics' in r['analysis']:
                topic = r['analysis']['topics'].get('dominant_topic', 'unknown')
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        stats['topic_distribution'] = topic_counts
        
        # Common keywords
        all_keywords = []
        for r in successful_results:
            if 'analysis' in r and 'keywords' in r['analysis']:
                keywords = r['analysis']['keywords'].get('keywords', [])
                all_keywords.extend([kw['keyword'] for kw in keywords[:5]])
        
        from collections import Counter
        keyword_counts = Counter(all_keywords)
        stats['common_keywords'] = keyword_counts.most_common(20)
        
        return stats
    
    def interactive_qa(self):
        """Start interactive Q&A session."""
        logger.info("Starting interactive Q&A session")
        print("\n" + "="*60)
        print("PolicyBot Interactive Q&A Session")
        print("="*60)
        print("Type 'quit' to exit, 'help' for commands")
        print()
        
        while True:
            try:
                question = input("Frage: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'help':
                    print("\nAvailable commands:")
                    print("- quit/exit/q: Exit the session")
                    print("- help: Show this help")
                    print("- stats: Show system statistics")
                    print("- Any other text will be treated as a question")
                    print()
                    continue
                elif question.lower() == 'stats':
                    stats = self.qa_system.get_system_stats()
                    print(f"\nSystem Statistics:")
                    print(f"- Vector store: {stats['vector_store']['total_chunks']} chunks")
                    print(f"- Model: {stats['qa_system']['model_name']}")
                    print(f"- Confidence threshold: {stats['qa_system']['confidence_threshold']}")
                    print()
                    continue
                
                if not question:
                    continue
                
                # Get answer
                answer = self.qa_system.ask_question(question)
                
                print(f"\nAntwort: {answer.answer}")
                print(f"Vertrauen: {answer.confidence:.2%}")
                print(f"Verarbeitungszeit: {answer.processing_time:.2f}s")
                
                if answer.sources:
                    print(f"\nQuellen ({len(answer.sources)}):")
                    for i, source in enumerate(answer.sources[:3], 1):
                        print(f"  {i}. Seite {source['page_number']} (Relevanz: {source['relevance_score']:.2%})")
                        print(f"     {source['text_preview']}")
                
                print("\n" + "-"*60)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                print(f"Fehler: {e}")
        
        print("\nInteractive session ended.")


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="PolicyBot: German Government PDF Analysis Pipeline")
    
    parser.add_argument(
        "--pdf_path", 
        type=str, 
        help="Path to single PDF file to process"
    )
    
    parser.add_argument(
        "--pdf_dir", 
        type=str, 
        help="Directory containing PDF files for batch processing"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Start interactive Q&A session"
    )
    
    parser.add_argument(
        "--save_config", 
        type=str, 
        help="Save current configuration to file"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = PolicyBotPipeline(args.config)
        
        # Save configuration if requested
        if args.save_config:
            save_config(args.save_config)
            print(f"Configuration saved to: {args.save_config}")
            return
        
        # Interactive mode
        if args.interactive:
            pipeline.interactive_qa()
            return
        
        # Single file processing
        if args.pdf_path:
            results = pipeline.process_document(args.pdf_path, args.output_dir)
            print(f"Document processed successfully. Results saved to: {args.output_dir}")
            return
        
        # Batch processing
        if args.pdf_dir:
            results = pipeline.batch_process(args.pdf_dir, args.output_dir)
            print(f"Batch processing completed: {results['processed_files']} successful, "
                  f"{results['failed_files']} failed")
            return
        
        # No action specified
        parser.print_help()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 