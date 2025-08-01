"""
PolicyBot Streamlit Web Application.

A web interface for processing German government PDFs with Q&A capabilities.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys
import os
import json
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import get_config
from src.pdf_processor import PDFProcessor
from src.vector_store import VectorStore
from src.langchain_qa import LangChainQA
from src.summarizer import DocumentSummarizer
from src.keyword_extractor import KeywordExtractor


# Page configuration
st.set_page_config(
    page_title="PolicyBot - German Government PDF Analysis",
    page_icon="🇩🇪",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_components():
    """Initialize all components with caching."""
    try:
        config = get_config()
        pdf_processor = PDFProcessor(config)
        vector_store = VectorStore(config)
        qa_system = LangChainQA(config, vector_store)
        summarizer = DocumentSummarizer(config)
        keyword_extractor = KeywordExtractor(config)
        
        return {
            'config': config,
            'pdf_processor': pdf_processor,
            'vector_store': vector_store,
            'qa_system': qa_system,
            'summarizer': summarizer,
            'keyword_extractor': keyword_extractor
        }
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None


def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🇩🇪 PolicyBot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Intelligent Analysis of German Government Documents</p>', unsafe_allow_html=True)
    
    # Initialize components
    components = initialize_components()
    if not components:
        st.error("Failed to initialize system components.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📄 Document Upload", "🤖 Q&A Interface", "📊 Analysis Dashboard"]
    )
    
    # Page routing
    if page == "📄 Document Upload":
        document_upload_page(components)
    elif page == "🤖 Q&A Interface":
        qa_interface_page(components)
    elif page == "📊 Analysis Dashboard":
        analysis_dashboard_page(components)


def document_upload_page(components):
    """Document upload and processing page."""
    st.markdown("## 📄 Document Upload & Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a German government PDF file",
        type=['pdf'],
        help="Upload a PDF document from govdata.de or other German government sources"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("temp") / uploaded_file.name
        temp_path.parent.mkdir(exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Processing options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            process_pdf = st.button("🔄 Process PDF", type="primary")
        
        with col2:
            extract_text = st.button("📝 Extract Text")
        
        with col3:
            analyze_doc = st.button("🔍 Analyze Document")
        
        if process_pdf:
            with st.spinner("Processing PDF document..."):
                try:
                    # Process the PDF
                    pdf_results = components['pdf_processor'].process_pdf(str(temp_path))
                    
                    # Store results in session state
                    st.session_state['pdf_results'] = pdf_results
                    st.session_state['document_name'] = uploaded_file.name
                    
                    # Display results
                    display_pdf_results(pdf_results)
                    
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
        
        elif extract_text:
            with st.spinner("Extracting text..."):
                try:
                    pdf_results = components['pdf_processor'].process_pdf(str(temp_path))
                    
                    st.markdown("### Extracted Text Preview")
                    text_preview = pdf_results['cleaned_text'][:2000] + "..." if len(pdf_results['cleaned_text']) > 2000 else pdf_results['cleaned_text']
                    st.text_area("Text Content", text_preview, height=300)
                    
                    # Document statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Characters", pdf_results['structure']['total_chars'])
                    with col2:
                        st.metric("Total Words", pdf_results['structure']['total_words'])
                    with col3:
                        st.metric("Total Sentences", pdf_results['structure']['total_sentences'])
                    with col4:
                        st.metric("Text Chunks", pdf_results['structure']['num_chunks'])
                    
                except Exception as e:
                    st.error(f"Error extracting text: {e}")
        
        elif analyze_doc:
            with st.spinner("Analyzing document..."):
                try:
                    # Process PDF first
                    pdf_results = components['pdf_processor'].process_pdf(str(temp_path))
                    
                    # Perform analysis
                    analysis_results = perform_document_analysis(components, pdf_results['cleaned_text'])
                    
                    # Store results
                    st.session_state['analysis_results'] = analysis_results
                    st.session_state['document_name'] = uploaded_file.name
                    
                    # Display analysis
                    display_analysis_results(analysis_results)
                    
                except Exception as e:
                    st.error(f"Error analyzing document: {e}")
        
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def perform_document_analysis(components, text):
    """Perform comprehensive document analysis."""
    results = {}
    
    # Add documents to vector store
    components['vector_store'].add_documents(components['pdf_processor']._create_chunks(text))
    
    # Generate summaries
    with st.spinner("Generating summaries..."):
        abstractive = components['summarizer'].summarize_document(text, summary_type="abstractive")
        extractive = components['summarizer'].summarize_document(text, summary_type="extractive")
        
        results['summaries'] = {
            'abstractive': abstractive,
            'extractive': extractive
        }
    
    # Extract keywords and classify topics
    with st.spinner("Extracting keywords and classifying topics..."):
        keywords = components['keyword_extractor'].extract_keywords(text)
        topics = components['keyword_extractor'].classify_topics(text)
        
        results['analysis'] = {
            'keywords': keywords,
            'topics': topics
        }
    
    return results


def display_pdf_results(pdf_results):
    """Display PDF processing results."""
    st.markdown("### 📋 Document Information")
    
    metadata = pdf_results['metadata']
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Title:** {metadata.title}")
        st.write(f"**Author:** {metadata.author}")
        st.write(f"**Pages:** {metadata.num_pages}")
        st.write(f"**Language:** {metadata.language}")
    
    with col2:
        st.write(f"**File Size:** {metadata.file_size / 1024:.1f} KB")
        st.write(f"**Created:** {metadata.creation_date}")
        st.write(f"**Modified:** {metadata.modification_date}")
        st.write(f"**Source:** {metadata.source}")
    
    st.markdown("### 📊 Document Statistics")
    structure = pdf_results['structure']
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Characters", f"{structure['total_chars']:,}")
    with col2:
        st.metric("Total Words", f"{structure['total_words']:,}")
    with col3:
        st.metric("Total Sentences", structure['total_sentences'])
    with col4:
        st.metric("Text Chunks", structure['num_chunks'])


def display_analysis_results(analysis_results):
    """Display document analysis results."""
    st.markdown("### 📝 Document Summaries")
    
    summaries = analysis_results['summaries']
    
    # Summary tabs
    tab1, tab2 = st.tabs(["Abstractive", "Extractive"])
    
    with tab1:
        if 'abstractive' in summaries:
            summary = summaries['abstractive']
            st.write(summary.summary)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", summary.length)
            with col2:
                st.metric("Compression", f"{summary.compression_ratio:.1%}")
            with col3:
                st.metric("Confidence", f"{summary.confidence:.1%}")
    
    with tab2:
        if 'extractive' in summaries:
            summary = summaries['extractive']
            st.write(summary.summary)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Length", summary.length)
            with col2:
                st.metric("Compression", f"{summary.compression_ratio:.1%}")
            with col3:
                st.metric("Confidence", f"{summary.confidence:.1%}")
    
    # Analysis results
    if 'analysis' in analysis_results:
        analysis = analysis_results['analysis']
        
        st.markdown("### 🔑 Keywords & Topics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Keywords:**")
            if 'keywords' in analysis:
                keywords = analysis['keywords'].keywords[:10]
                for i, kw in enumerate(keywords, 1):
                    st.write(f"{i}. {kw['keyword']} (Score: {kw['score']:.3f})")
        
        with col2:
            st.markdown("**Topic Classification:**")
            if 'topics' in analysis:
                topics = analysis['topics']
                st.write(f"**Dominant Topic:** {topics.dominant_topic}")
                st.write(f"**Confidence:** {topics.confidence:.1%}")
                
                # Topic distribution chart
                if topics.topic_distribution:
                    topic_df = pd.DataFrame(list(topics.topic_distribution.items()), 
                                          columns=['Topic', 'Confidence'])
                    fig = px.bar(topic_df, x='Topic', y='Confidence', 
                               title="Topic Distribution")
                    st.plotly_chart(fig, use_container_width=True)


def qa_interface_page(components):
    """Q&A interface page."""
    st.markdown("## 🤖 Question & Answer Interface")
    
    # Check if document is loaded
    if 'pdf_results' not in st.session_state:
        st.warning("Please upload and process a document first in the Document Upload page.")
        return
    
    # Q&A interface
    st.markdown("### Ask Questions About the Document")
    
    # Question input
    question = st.text_input(
        "Enter your question (in German or English):",
        placeholder="e.g., Was ist das Hauptthema dieses Dokuments?"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary")
    
    with col2:
        include_sources = st.checkbox("Include source references", value=True)
    
    if ask_button and question:
        with st.spinner("Processing your question..."):
            try:
                # Get answer
                answer = components['qa_system'].ask_question(question, include_sources)
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(f"<div class='metric-card'>{answer.answer}</div>", unsafe_allow_html=True)
                
                # Answer metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{answer.confidence:.1%}")
                with col2:
                    st.metric("Processing Time", f"{answer.processing_time:.2f}s")
                with col3:
                    st.metric("Sources", len(answer.sources))
                
                # Display sources
                if answer.sources:
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(answer.sources, 1):
                        with st.expander(f"Source {i} - Page {source['page_number']} (Relevanz: {source['relevance_score']:.1%})"):
                            st.write(source['text_preview'])
                
            except Exception as e:
                st.error(f"Error processing question: {e}")
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "Was ist das Hauptthema dieses Dokuments?",
        "Welche Ziele werden in diesem Dokument genannt?",
        "Welche Maßnahmen werden vorgeschlagen?",
        "Wer ist für die Umsetzung verantwortlich?",
        "Welche Zeitpläne werden erwähnt?"
    ]
    
    cols = st.columns(3)
    for i, sample_q in enumerate(sample_questions):
        with cols[i % 3]:
            if st.button(sample_q, key=f"sample_{i}"):
                st.session_state['question'] = sample_q
                st.rerun()


def analysis_dashboard_page(components):
    """Analysis dashboard page."""
    st.markdown("## 📊 Analysis Dashboard")
    
    # Check if analysis results are available
    if 'analysis_results' not in st.session_state:
        st.warning("Please analyze a document first in the Document Upload page.")
        return
    
    analysis_results = st.session_state['analysis_results']
    
    # System statistics
    st.markdown("### 🖥️ System Statistics")
    
    try:
        stats = components['qa_system'].get_system_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Vector Store Chunks", stats['vector_store']['total_chunks'])
        with col2:
            st.metric("Model", stats['qa_system']['model_name'].split('/')[-1])
        with col3:
            st.metric("Confidence Threshold", f"{stats['qa_system']['confidence_threshold']:.1%}")
        with col4:
            st.metric("Similarity Metric", stats['retrieval']['similarity_metric'])
    
    except Exception as e:
        st.warning(f"Could not load system statistics: {e}")
    
    # Analysis results overview
    if 'analysis' in analysis_results:
        analysis = analysis_results['analysis']
        
        st.markdown("### 📈 Analysis Overview")
        
        # Keywords analysis
        if 'keywords' in analysis:
            keywords = analysis['keywords']
            st.markdown("#### 🔑 Keywords Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Keywords", keywords.total_keywords)
                st.metric("Processing Time", f"{keywords.processing_time:.2f}s")
            
            with col2:
                # Keyword scores distribution
                if keywords.keywords:
                    scores = [kw['score'] for kw in keywords.keywords]
                    fig = px.histogram(x=scores, title="Keyword Score Distribution")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Topic analysis
        if 'topics' in analysis:
            topics = analysis['topics']
            st.markdown("#### 🏷️ Topic Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Dominant Topic", topics.dominant_topic)
                st.metric("Confidence", f"{topics.confidence:.1%}")
            
            with col2:
                # Topic distribution pie chart
                if topics.topic_distribution:
                    topic_df = pd.DataFrame(list(topics.topic_distribution.items()), 
                                          columns=['Topic', 'Confidence'])
                    fig = px.pie(topic_df, values='Confidence', names='Topic', 
                               title="Topic Distribution")
                    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main() 