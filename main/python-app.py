import sys
import streamlit as st
import pandas as pd
import random
import io
# from openai import OpenAI # Not directly used for OpenRouter
import json
from io import BytesIO
from typing import List, Optional, Dict, Any, Tuple, AsyncGenerator
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pdfminer.high_level import extract_text as pdf_extract_text
import tempfile
import time
# import openai # Not directly used for OpenRouter
import requests
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import asyncio
import aiohttp
from functools import lru_cache
import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
VALID_TRANSCRIPT_TYPES = [".txt", ".docx", ".pdf"]
VALID_FRAMEWORK_TYPES = ["json", "txt"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat:free"
MAX_TOKENS = 4000
TEMPERATURE = 0.7
CHUNK_SIZE = 1500  # Reduced for better context management
OVERLAP = 200
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
CHUNK_READ_SIZE = 1024 * 1024  # 1 MB chunks for streaming

# Theme colors for highlighting
THEME_COLORS = {
    "Learning Experience": "#FFB6C1",  # Light pink
    "User Interface": "#98FB98",       # Light green
    "Technical Issues": "#87CEEB",     # Sky blue
    "Feedback": "#DDA0DD",             # Plum
    "General": "#F0E68C"               # Khaki
}

# Get API key from secrets
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Define the headers for the API request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/baha-ti/Qualinsight-AI",
    "X-Title": "Qualinsight AI"
}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def extract_text_from_txt(file) -> str:
    """Extract text from TXT file with caching."""
    logger.info("Extracting text from TXT file (cache miss)")
    return file.getvalue().decode('utf-8')

@st.cache_data(ttl=3600)
def extract_text_from_docx(file) -> str:
    """Extract text from DOCX file with caching."""
    logger.info("Extracting text from DOCX file (cache miss)")
    doc = Document(file)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

@st.cache_data(ttl=3600)
def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file with caching."""
    logger.info("Extracting text from PDF file (cache miss)")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file.flush()
        text = pdf_extract_text(tmp_file.name)
    os.unlink(tmp_file.name)
    return text

@st.cache_data(ttl=3600)
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping chunks with context preservation and caching."""
    logger.info("Chunking text (cache miss)")
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        # Get chunk of words
        end = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        
        # Add context from previous chunk if not first chunk
        if start > 0:
            context_start = max(0, start - overlap)
            context = ' '.join(words[context_start:start])
            chunk = f"{context}\n\n{chunk}"
        
        # Add context from next chunk if not last chunk
        if end < len(words):
            context_end = min(len(words), end + overlap)
            context = ' '.join(words[end:context_end])
            chunk = f"{chunk}\n\n{context}"
        
        chunks.append(chunk)
        start = end
    
    return chunks

async def stream_file(file, chunk_size: int = CHUNK_READ_SIZE) -> AsyncGenerator[bytes, None]:
    """Stream file in chunks."""
    while True:
        chunk = file.read(chunk_size)
        if not chunk:
            break
        yield chunk

async def process_large_file(file) -> str:
    """Process large files in chunks."""
    content = []
    async for chunk in stream_file(file):
        content.append(chunk)
    return b''.join(content).decode('utf-8')

async def get_ai_response_async(messages: List[Dict], stream: bool = True) -> AsyncGenerator[str, None]:
    """Get response from DeepSeek API asynchronously."""
    logger.info("Starting async API request")
    start_time = time.time()
    
    data = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": stream
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, json=data, headers=headers) as response:
            if response.status == 200:
                if stream:
                    async for line in response.content:
                        if line:
                            try:
                                json_response = json.loads(line.decode('utf-8').replace('data: ', ''))
                                if 'choices' in json_response and len(json_response['choices']) > 0:
                                    content = json_response['choices'][0].get('delta', {}).get('content', '')
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
                else:
                    result = await response.json()
                    yield result['choices'][0]['message']['content']
                
                logger.info(f"API request completed in {time.time() - start_time:.2f} seconds")
            else:
                error_text = await response.text()
                logger.error(f"API Error: {response.status} - {error_text}")
                raise Exception(f"API Error: {response.status} - {error_text}")

def get_ai_response(messages: List[Dict], stream: bool = True):
    """Synchronous wrapper for async AI response."""
    if stream:
        return asyncio.run(get_ai_response_async(messages, stream))
    else:
        return asyncio.run(get_ai_response_async(messages, stream))

def process_transcript(uploaded_file) -> Optional[List[str]]:
    """Process uploaded transcript files with size limits and streaming."""
    start_time = time.time()
    logger.info(f"Processing file: {uploaded_file.name} ({uploaded_file.size/1024/1024:.2f}MB)")
    
    if uploaded_file.size > MAX_FILE_SIZE:
        st.warning(f"File size exceeds {MAX_FILE_SIZE/1024/1024}MB limit. Processing in chunks...")
        try:
            # Process large file asynchronously
            text_content = asyncio.run(process_large_file(uploaded_file))
            logger.info(f"Large file processed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error processing large file: {str(e)}")
            st.error(f"Error processing large file: {str(e)}")
            return None
    else:
        file_type = uploaded_file.name.split('.')[-1].lower()
        try:
            if file_type == "txt":
                text_content = extract_text_from_txt(uploaded_file)
            elif file_type == "docx":
                text_content = extract_text_from_docx(uploaded_file)
            elif file_type == "pdf":
                text_content = extract_text_from_pdf(uploaded_file)
            else:
                st.error(f"Unsupported file type: {file_type}")
                return None
            logger.info(f"File processed in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            st.error(f"Error extracting text: {str(e)}")
            return None

    if text_content:
        # Chunk the text with caching
        chunks = chunk_text(text_content)
        logger.info(f"Text chunked into {len(chunks)} chunks")
        return chunks
    return None

def highlight_text(text, theme, color):
    """Wrap text in HTML mark tag with specified color"""
    return f'<mark style="background-color: {color}">{text}</mark>'

# Initialize session state
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = {}

# Knowledge Base File Operations
def load_knowledge_base() -> Dict[str, Dict[str, Any]]:
    """Load knowledge base from JSON file"""
    kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
    if os.path.exists(kb_path):
        with open(kb_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_knowledge_base(kb: Dict[str, Dict[str, Any]]):
    """Save knowledge base to JSON file"""
    kb_path = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
    with open(kb_path, 'w', encoding='utf-8') as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

# Load knowledge base at startup
st.session_state.knowledge_base = load_knowledge_base()

# --- Title ---
st.title("QualInsight AI - Qualitative Research Assistant")

# Create tabs for different sections
tab1, tab2, tab3, tab4 = st.tabs(["Input", "Analysis", "Results", "Export"])

with tab1:
    st.header("Step 1: Input")
    
    # Framework Upload (for deductive analysis)
    if st.session_state.analysis_mode == "Deductive":
        st.subheader("Coding Framework")
        framework_file = st.file_uploader("Upload Framework (JSON)", type=["json"])
        
        if framework_file:
            try:
                framework = load_framework(framework_file)
                with st.sidebar:
                    st.subheader("Framework Preview")
                    st.write(f"**Name:** {framework['name']}")
                    st.write(f"**Description:** {framework['description']}")
                    
                    st.write("**Categories and Codes:**")
                    for category in framework['categories']:
                        with st.expander(category['name']):
                            for code in category['codes']:
                                st.write(f"- {code['name']}: {code['description']}")
                
                st.session_state.framework = framework
            except ValueError as e:
                st.error(str(e))
    
    st.subheader("Upload Transcript")
    input_method = st.radio("Choose input method:", ["Upload File", "Paste Text"], key="input_method_radio")

    transcript_text = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload transcript (.txt, .docx, or .pdf)",
            type=VALID_TRANSCRIPT_TYPES,
            key="transcript_uploader"
        )
        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.warning(f"File size ({uploaded_file.size/1024/1024:.1f}MB) exceeds {MAX_FILE_SIZE/1024/1024}MB limit. Processing may take longer.")
            transcript_text = process_transcript(uploaded_file)
    else:
        pasted_text = st.text_area("Paste your transcript here:", height=300, key="transcript_text_area")
        if pasted_text:
            transcript_text = [pasted_text]

    # Research Questions
    research_questions = st.text_area(
        "Your Research Questions:",
        "",
        height=100,
        key="research_questions_text_area",
        help="Enter the research questions guiding your analysis."
    )

    # Analysis Mode Selection
    analysis_mode = st.radio(
        "Select Analysis Approach:",
        ("Inductive", "Deductive"),
        key="analysis_mode_radio",
        index=0  # Default to Inductive
    )
    st.session_state.analysis_mode = analysis_mode

with tab2:
    st.header("Step 2: Analysis")
    
    # Analysis Parameters
    with st.expander("Analysis Parameters", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.1,
                              help="Higher values make output more random, lower values more focused")
        max_tokens = st.slider("Max Tokens", 1000, 8000, MAX_TOKENS, 1000,
                             help="Maximum number of tokens to generate")
        model = st.selectbox("Model", ["deepseek/deepseek-chat:free", "deepseek/deepseek-chat:paid"],
                           help="Select the model to use for analysis")
        
        # Chunking parameters
        st.subheader("Chunking Settings")
        chunk_size = st.slider("Chunk Size", 500, 3000, CHUNK_SIZE, 100,
                             help="Size of text chunks for analysis")
        overlap = st.slider("Overlap", 50, 500, OVERLAP, 50,
                          help="Number of words to overlap between chunks")

    # --- Trigger Analysis Button ---
    if st.button("Start Analysis", key="start_analysis_button"):
        if not transcript_text:
            st.error("Please upload or paste a transcript to start analysis.")
        elif not research_questions:
            st.error("Please enter your research questions to start analysis.")
        else:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process the chunks
            all_results = []
            for i, chunk in enumerate(transcript_text):
                try:
                    status_text.text(f"Processing chunk {i+1} of {len(transcript_text)}...")
                    progress_bar.progress((i + 1) / len(transcript_text))
                    
                    # First stage prompt - focus on identifying codes and organizing by respondent
                    system_prompt = """You are an AI assistant analyzing qualitative research transcripts.
                    For this first stage, focus on:
                    1. Identifying key codes/themes in the text
                    2. Organizing codes by respondent
                    3. Providing brief notes for each code

                    IMPORTANT: Your response MUST be a plain text string, with each code entry separated by `###CODE_ENTRY###`. Each entry must contain the following fields, delimited by `|||`:
                    - RESPONDENT: <respondent identifier>
                    - TEXT: <exact quote from transcript>
                    - CODE: <code/theme label>
                    - NOTES: <brief explanation>

                    Example response format:
                    RESPONDENT: P1 ||| TEXT: The interface was very intuitive ||| CODE: UI Clarity ||| NOTES: Positive feedback about interface design###CODE_ENTRY###
                    """
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Research Questions: {research_questions}\n\nTranscript Chunk:\n{chunk}"}
                    ]
                    
                    response = get_ai_response(messages, stream=True)
                    if response:
                        # Process streaming response
                        full_response = ""
                        response_container = st.empty()
                        
                        for line in response.iter_lines():
                            if line:
                                try:
                                    json_response = json.loads(line.decode('utf-8').replace('data: ', ''))
                                    if 'choices' in json_response and len(json_response['choices']) > 0:
                                        content = json_response['choices'][0].get('delta', {}).get('content', '')
                                        if content:
                                            full_response += content
                                            response_container.markdown(full_response)
                                except json.JSONDecodeError:
                                    continue
                        
                        # Process the full response
                        try:
                            code_entries = full_response.split('###CODE_ENTRY###')
                            chunk_results = []
                            for entry in code_entries:
                                if entry.strip():
                                    parts = entry.split('|||')
                                    if len(parts) == 4:
                                        result = {
                                            'respondent': parts[0].replace('RESPONDENT:', '').strip(),
                                            'text': parts[1].replace('TEXT:', '').strip(),
                                            'code': parts[2].replace('CODE:', '').strip(),
                                            'notes': parts[3].replace('NOTES:', '').strip()
                                        }
                                        chunk_results.append(result)
                            all_results.append(chunk_results)
                        except Exception as e:
                            st.error(f"Failed to parse AI response for chunk {i+1}. Error: {str(e)}")
                            st.text("Raw response:")
                            st.text(full_response)
                            continue
                except Exception as e:
                    st.error(f"Error processing chunk {i+1}: {str(e)}")
            
            # Merge results from all chunks
            if all_results:
                merged_results = merge_coded_chunks(all_results)
                st.session_state.coded_results = merged_results
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

with tab3:
    st.header("Step 3: Results")
    
    if 'coded_results' in st.session_state:
        # Create DataFrame for initial coding
        df_initial = pd.DataFrame(st.session_state.coded_results)
        
        # Display results in expandable sections
        with st.expander("Coding by Respondent", expanded=True):
            for respondent in df_initial['respondent'].unique():
                st.write(f"**Respondent: {respondent}**")
                respondent_df = df_initial[df_initial['respondent'] == respondent]
                
                # Create editable dataframe
                edited_df = st.data_editor(
                    respondent_df[['text', 'code', 'notes']],
                    key=f"editor_{respondent}",
                    num_rows="dynamic",
                    use_container_width=True
                )
                
                # Update the original dataframe with edited values
                if edited_df is not None:
                    df_initial.update(edited_df)
                    st.session_state.coded_results = df_initial.to_dict('records')
                
                st.write("---")
        
        # Theme Development
        with st.expander("Theme Development", expanded=False):
            if st.button("Proceed to Theme Development", key="theme_development"):
                try:
                    with st.spinner("Developing themes and subthemes..."):
                        # Second stage prompt - focus on theme development
                        system_prompt = """You are an AI assistant developing themes from coded transcript segments.
                        For this second stage, focus on:
                        1. Grouping related codes into themes
                        2. Identifying subthemes within each theme
                        3. Providing evidence from the transcript

                        IMPORTANT: Your response MUST be a plain text string, with each theme entry separated by `###THEME_ENTRY###`. Each entry must contain the following fields, delimited by `|||`:
                        - THEME: <main theme>
                        - SUBTHEME: <subtheme within the main theme>
                        - CODES: <list of related codes, comma-separated>
                        - EVIDENCE: <list of relevant quotes, comma-separated>
                        - EXPLANATION: <brief explanation of this theme>

                        Example response format:
                        THEME: Learning Experience ||| SUBTHEME: Instructional Clarity ||| CODES: Clear Instructions, Understanding ||| EVIDENCE: The instructions were very clear, I understood what to do, it was "spot on" ||| EXPLANATION: Participants found the instructions clear and easy to follow###THEME_ENTRY###
                        """
                        
                        # Convert initial coding to string for analysis
                        coding_summary_str = df_initial.to_string(index=False)
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Research Questions: {research_questions}\n\nCoded Transcript:\n{coding_summary_str}"}
                        ]
                        
                        response = get_ai_response(messages)
                        if response:
                            try:
                                theme_entries = response.split('###THEME_ENTRY###')
                                theme_results = []
                                
                                for entry in theme_entries:
                                    if entry.strip():
                                        parts = entry.split('|||')
                                        if len(parts) == 5:
                                            theme_result = {
                                                'theme': parts[0].replace('THEME:', '').strip(),
                                                'subtheme': parts[1].replace('SUBTHEME:', '').strip(),
                                                'codes': [code.strip() for code in parts[2].replace('CODES:', '').split(',')],
                                                'evidence': [quote.strip().strip('"\'') for quote in parts[3].replace('EVIDENCE:', '').split(',')],
                                                'explanation': parts[4].replace('EXPLANATION:', '').strip()
                                            }
                                            theme_results.append(theme_result)
                                
                                df_themes = pd.DataFrame(theme_results)
                                st.session_state.theme_results = df_themes
                                st.dataframe(df_themes)
                                
                                # Create highlighted transcript
                                with st.expander("Highlighted Transcript", expanded=False):
                                    highlighted_text = transcript_text[0]  # Assuming single transcript
                                    
                                    # Apply highlights
                                    for _, row in df_themes.iterrows():
                                        for quote in row['evidence']:
                                            if quote in highlighted_text:
                                                theme = row['theme']
                                                color = THEME_COLORS.get(theme, THEME_COLORS["General"])
                                                highlighted_text = highlighted_text.replace(
                                                    quote,
                                                    highlight_text(quote, theme, color)
                                                )
                                    
                                    st.markdown(highlighted_text, unsafe_allow_html=True)
                                
                            except Exception as e:
                                st.error(f"Failed to parse theme development response. Error: {str(e)}")
                                st.text("Raw response:")
                                st.text(response)
                except Exception as e:
                    st.error(f"Error in theme development: {str(e)}")
        
        # Inter-coder Reliability
        with st.expander("Inter-coder Reliability", expanded=False):
            st.write("Upload coding results from multiple coders to calculate reliability metrics")
            
            uploaded_files = st.file_uploader(
                "Upload coding results (CSV files)",
                type=["csv"],
                accept_multiple_files=True
            )
            
            if uploaded_files and len(uploaded_files) >= 2:
                try:
                    coders_data = []
                    for file in uploaded_files:
                        df = pd.read_csv(file)
                        coders_data.append(df)
                    
                    reliability_scores = calculate_intercoder_reliability(coders_data)
                    
                    st.write("### Reliability Scores")
                    for pair, score in reliability_scores.items():
                        st.write(f"{pair}: {score:.3f}")
                    
                except Exception as e:
                    st.error(f"Error calculating reliability: {str(e)}")

with tab4:
    st.header("Step 4: Export")
    
    if 'coded_results' in st.session_state and 'theme_results' in st.session_state:
        # Export Options
        st.subheader("Export Options")
        
        # Export initial coding
        csv_initial = pd.DataFrame(st.session_state.coded_results).to_csv(index=False)
        st.download_button(
            "Download Initial Coding (CSV)",
            csv_initial,
            "initial_coding.csv",
            "text/csv",
            key="download_initial_csv"
        )
        
        # Export themes
        csv_themes = st.session_state.theme_results.to_csv(index=False)
        st.download_button(
            "Download Themes (CSV)",
            csv_themes,
            "themes.csv",
            "text/csv",
            key="download_themes_csv"
        )
        
        # Export full report
        if st.button("Generate Full Report"):
            try:
                # Create Word document
                doc = Document()
                doc.add_heading("Qualitative Research Analysis Report", 0)
                
                # Add research questions
                doc.add_heading("Research Questions", level=1)
                doc.add_paragraph(research_questions)
                
                # Add initial coding
                doc.add_heading("Initial Coding", level=1)
                for respondent in pd.DataFrame(st.session_state.coded_results)['respondent'].unique():
                    doc.add_heading(f"Respondent: {respondent}", level=2)
                    respondent_df = pd.DataFrame(st.session_state.coded_results)
                    respondent_df = respondent_df[respondent_df['respondent'] == respondent]
                    for _, row in respondent_df.iterrows():
                        doc.add_paragraph(f"Code: {row['code']}")
                        doc.add_paragraph(f"Text: {row['text']}")
                        doc.add_paragraph(f"Notes: {row['notes']}")
                        doc.add_paragraph("---")
                
                # Add themes
                doc.add_heading("Theme Development", level=1)
                for _, row in st.session_state.theme_results.iterrows():
                    doc.add_heading(f"Theme: {row['theme']}", level=2)
                    doc.add_paragraph(f"Subtheme: {row['subtheme']}")
                    doc.add_paragraph(f"Codes: {', '.join(row['codes'])}")
                    doc.add_paragraph(f"Evidence: {', '.join(row['evidence'])}")
                    doc.add_paragraph(f"Explanation: {row['explanation']}")
                    doc.add_paragraph("---")
                
                # Add highlighted transcript
                doc.add_heading("Highlighted Transcript", level=1)
                doc.add_paragraph(highlighted_text)
                
                doc_stream = BytesIO()
                doc.save(doc_stream)
                doc_stream.seek(0)
                st.download_button(
                    "Download Full Report (Word)",
                    doc_stream,
                    "analysis_report.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_word"
                )
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")

def merge_coded_chunks(chunks: List[Dict]) -> List[Dict]:
    """Merge coded chunks while handling overlapping segments."""
    merged = []
    seen_texts = set()
    
    for chunk in chunks:
        for entry in chunk:
            # Skip if we've seen this exact text before
            if entry['text'] in seen_texts:
                continue
            
            # Check for overlapping segments
            is_overlap = False
            for existing in merged:
                if entry['text'] in existing['text'] or existing['text'] in entry['text']:
                    # Keep the longer segment
                    if len(entry['text']) > len(existing['text']):
                        merged.remove(existing)
                        merged.append(entry)
                    is_overlap = True
                    break
            
            if not is_overlap:
                merged.append(entry)
                seen_texts.add(entry['text'])
    
    return merged

def calculate_intercoder_reliability(coders_data: List[pd.DataFrame]) -> Dict[str, float]:
    """Calculate Cohen's Kappa and Fleiss' Kappa for multiple coders."""
    if len(coders_data) < 2:
        return {"error": "Need at least 2 coders for reliability analysis"}
    
    # Get unique segments across all coders
    all_segments = set()
    for df in coders_data:
        all_segments.update(df['text'].tolist())
    
    # Create coding matrices
    segment_codes = {segment: [] for segment in all_segments}
    
    for df in coders_data:
        for _, row in df.iterrows():
            segment_codes[row['text']].append(row['code'])
    
    # Calculate pairwise Cohen's Kappa
    kappa_scores = {}
    for i, j in combinations(range(len(coders_data)), 2):
        coder1_codes = [codes[i] if i < len(codes) else None for codes in segment_codes.values()]
        coder2_codes = [codes[j] if j < len(codes) else None for codes in segment_codes.values()]
        
        # Filter out None values
        valid_pairs = [(c1, c2) for c1, c2 in zip(coder1_codes, coder2_codes) if c1 is not None and c2 is not None]
        if valid_pairs:
            c1_codes, c2_codes = zip(*valid_pairs)
            kappa = cohen_kappa_score(c1_codes, c2_codes)
            kappa_scores[f"Coders {i+1}-{j+1}"] = kappa
    
    return kappa_scores

def load_framework(file) -> Dict:
    """Load and validate a coding framework from JSON file."""
    try:
        framework = json.load(file)
        required_keys = ['name', 'description', 'codes', 'categories']
        if not all(key in framework for key in required_keys):
            raise ValueError("Framework missing required keys")
        return framework
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")
    except Exception as e:
        raise ValueError(f"Error loading framework: {str(e)}")

def ai_generate_codes(text: str, mode: str, rq: str, framework: Optional[str] = None) -> List[dict]:
    """Generates initial codes or themes using the AI."""
    # This function is not used directly anymore, but is kept for reference or future use.
    # The prompt building and API call logic is now directly in the main app flow for more control.
    pass

# Add debug information to the UI
if st.sidebar.checkbox("Show Debug Information"):
    st.sidebar.write("### Debug Information")
    st.sidebar.write(f"Cache TTL: 3600 seconds")
    st.sidebar.write(f"Max File Size: {MAX_FILE_SIZE/1024/1024}MB")
    st.sidebar.write(f"Chunk Size: {CHUNK_SIZE} words")
    st.sidebar.write(f"Chunk Overlap: {OVERLAP} words")
    
    if 'analysis_start_time' in st.session_state:
        st.sidebar.write(f"Last Analysis Duration: {time.time() - st.session_state.analysis_start_time:.2f} seconds")