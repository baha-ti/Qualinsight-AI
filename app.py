import sys
import streamlit as st
import pandas as pd
import random
import io
from openai import OpenAI
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
import requests
import os
import numpy as np
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import asyncio
import aiohttp
from functools import lru_cache, wraps
import hashlib
import logging
import fitz  # PyMuPDF
import tiktoken
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import traceback
from typing import Union, TypeVar, Callable
import threading
from queue import Queue
import uuid
import weakref
from PIL import Image
import pytesseract
import re
from pdfminer.layout import LAParams

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Streamlit page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Qualinsight AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 1000  # Maximum number of cached items
CACHE_LOCK = threading.Lock()

# Thread pool for CPU-bound tasks
THREAD_POOL = ThreadPoolExecutor(max_workers=4)

# Global cache for expensive operations
_global_cache = {}  # Changed from WeakValueDictionary to regular dict

def generate_cache_key(*args, **kwargs) -> str:
    """Generate a unique cache key from function arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()

def cache_result(ttl: int = CACHE_TTL, max_size: int = MAX_CACHE_SIZE):
    """Decorator for caching function results with TTL and size limits."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = generate_cache_key(func.__name__, *args, **kwargs)
            
            with CACHE_LOCK:
                # Check if result is in cache and not expired
                if cache_key in _global_cache:
                    result, timestamp = _global_cache[cache_key]
                    if time.time() - timestamp < ttl:
                        return result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                
                # Clean up old entries if cache is too large
                if len(_global_cache) >= max_size:
                    # Remove oldest entries
                    sorted_entries = sorted(_global_cache.items(), key=lambda x: x[1][1])
                    for key, _ in sorted_entries[:max_size//2]:  # Remove half of the entries
                        del _global_cache[key]
                
                _global_cache[cache_key] = (result, time.time())
                return result
        return wrapper
    return decorator

class AsyncTaskManager:
    """Manages asynchronous tasks and their results using a single asyncio event loop in a dedicated thread."""
    def __init__(self):
        self.tasks = {}
        self.results = {}
        self.progress_queues: Dict[str, Queue] = {}
        self.lock = threading.Lock()
        self.loop = None
        self.thread = None

        # Start the asyncio event loop in a dedicated thread
        self._start_event_loop_thread()

    def _start_event_loop_thread(self):
        """Starts a dedicated thread to run the asyncio event loop."""
        if self.thread is None or not self.thread.is_alive():
            self.loop = asyncio.new_event_loop()
            self.thread = threading.Thread(target=self._run_event_loop, args=(self.loop,), daemon=True)
            self.thread.start()
            logger.info("Started dedicated asyncio event loop thread.")

    def _run_event_loop(self, loop):
        """Target function for the dedicated event loop thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def start_task(self, task_id: str, coro, progress_queue: Queue):
        """Submit an asynchronous task to the dedicated event loop."""
        if self.loop is None or not self.thread.is_alive():
            self._start_event_loop_thread() # Re-start if thread died

        async def wrapped_task():
            try:
                result = await coro
                with self.lock:
                    self.results[task_id] = result
                    progress_queue.put(("complete", result)) # Signal completion
                return result
            except Exception as e:
                with self.lock:
                    self.results[task_id] = e
                    progress_queue.put(("error", str(e))) # Signal error
                raise
        
        # Submit the coroutine to the event loop in the dedicated thread
        asyncio.run_coroutine_threadsafe(wrapped_task(), self.loop)
        self.progress_queues[task_id] = progress_queue
        logger.debug(f"Task {task_id} submitted to dedicated event loop.")
    
    def get_result(self, task_id: str) -> Optional[Any]:
        """Get the result of a task if it's complete."""
        with self.lock:
            if task_id in self.results:
                result = self.results[task_id]
                if isinstance(result, Exception):
                    raise result
                return result
            return None
    
    def get_progress_queue(self, task_id: str) -> Optional[Queue]:
        """Get the progress queue for a specific task."""
        with self.lock:
            return self.progress_queues.get(task_id)

    def is_complete(self, task_id: str) -> bool:
        """Check if a task is complete."""
        with self.lock:
            return task_id in self.results
    
    def cleanup(self, task_id: str):
        """Clean up a completed task."""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].cancel() # Cancel the asyncio task
                del self.tasks[task_id]
            if task_id in self.results:
                del self.results[task_id]
            if task_id in self.progress_queues:
                del self.progress_queues[task_id]
            logger.debug(f"Task {task_id} cleaned up from AsyncTaskManager.")

# Initialize task manager
task_manager = AsyncTaskManager()

# Initialize session state for task tracking
if "active_tasks" not in st.session_state:
    st.session_state.active_tasks = set()

# Initialize analysis timing variables
if "analysis_start_time" not in st.session_state:
    st.session_state.analysis_start_time = None
if "analysis_duration" not in st.session_state:
    st.session_state.analysis_duration = None

# Initialize analysis mode and framework
if "analysis_mode" not in st.session_state:
    st.session_state.analysis_mode = "Thematic Analysis" # Default mode
if "framework" not in st.session_state:
    st.session_state.framework = None # Default framework (no framework)

# Custom error types
class QualinsightError(Exception):
    """Base exception class for Qualinsight AI."""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

class AIResponseError(QualinsightError):
    """Raised when AI response is invalid or malformed."""
    pass

class FileProcessingError(QualinsightError):
    """Raised when file processing fails."""
    pass

class ValidationError(QualinsightError):
    """Raised when input validation fails."""
    pass

# Type variable for generic function return type
T = TypeVar('T')

def handle_errors(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for handling errors in a user-friendly way."""
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except QualinsightError as e:
            st.error(f"Error: {e.message}")
            if e.details:
                with st.expander("Technical Details"):
                    st.code(e.details)
            return None
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            st.error("An unexpected error occurred. Please try again or contact support.")
            with st.expander("Technical Details"):
                st.code(traceback.format_exc())
            return None
    return wrapper

def validate_ai_response(response: str, expected_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Validate AI response against expected schema."""
    try:
        # Try to parse JSON response
        if isinstance(response, str):
            data = json.loads(response)
        else:
            data = response

        # Validate required fields
        for key, expected_type in expected_schema.items():
            if key not in data:
                raise AIResponseError(f"Missing required field: {key}")
            if not isinstance(data[key], expected_type):
                raise AIResponseError(f"Invalid type for field {key}. Expected {expected_type}, got {type(data[key])}")
        return data
    except json.JSONDecodeError:
        raise AIResponseError("Invalid JSON response from AI.", traceback.format_exc())
    except AIResponseError:
        raise # Re-raise custom AIResponseError
    except Exception as e:
        raise AIResponseError(f"Error validating AI response: {e}", traceback.format_exc())

# Constants
VALID_TRANSCRIPT_TYPES = ["txt", "docx", "pdf"]
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
CHUNK_SIZE = 500  # words
OVERLAP = 50  # words
CHUNK_READ_SIZE = 8192 # Bytes for file streaming

# OpenRouter API configuration
API_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o"  # Default model

# Theme colors for highlighting
THEME_COLORS = {
    "Theme 1": "#FFADAD", # Light Red
    "Theme 2": "#FFD6A5", # Light Orange
    "Theme 3": "#FDFFB6", # Light Yellow
    "Theme 4": "#CAFFBF", # Light Green
    "Theme 5": "#9BF6FF", # Light Cyan
    "Theme 6": "#A0C4FF", # Light Blue
    "Theme 7": "#BDB2FF", # Light Purple
    "Theme 8": "#FFC6FF", # Light Pink
    "Theme 9": "#FFFFFC", # Off-white
    "Theme 10": "#E5E5E5", # Light Gray
}

# OpenAI client for API key validation (only for local validation, not for OpenRouter calls)
# Assuming OPENAI_API_KEY from .streamlit/secrets.toml or env variable
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def show_error_page(error: Exception):
    st.error("An error occurred:")
    st.exception(error)
    if isinstance(error, QualinsightError) and error.details:
        with st.expander("Technical Details"):
            st.code(error.details)
    st.info("Please try again or contact support.")

def validate_api_key(api_key: str) -> bool:
    """Validate the OpenAI API key format."""
    return isinstance(api_key, str) and api_key.strip() != "" and api_key.startswith("sk-")

def get_api_key() -> Optional[str]:
    """Get the OpenAI API key from Streamlit session state, environment variables, or Streamlit secrets."""
    api_key = None
    
    # 1. Try to get API key from Streamlit session state (for in-browser input)
    if "openai_api_key_input" in st.session_state and st.session_state.openai_api_key_input:
        api_key = st.session_state.openai_api_key_input
        logger.info("API key found in Streamlit session state (in-browser input).")
    # 2. Try to get API key from Streamlit secrets
    elif "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        logger.info("API key found in Streamlit secrets.")
    # 3. Try to get API key from environment variables
    elif "OPENAI_API_KEY" in os.environ:
        api_key = os.environ.get("OPENAI_API_KEY")
        logger.info("API key found in environment variables.")
    else:
        logger.warning("OPENAI_API_KEY not found in Streamlit session state, secrets, or environment variables.")
    
    if not validate_api_key(api_key):
        logger.warning("Invalid OpenAI API key format.")
        return None
    
    return api_key

def show_api_key_guidance():
    """Show guidance for setting up the OpenAI API key (simplified for in-browser input)."""
    st.info("""
    To use this application, please enter your OpenAI API key above or configure it via `.streamlit/secrets.toml` or environment variables.
    You can obtain your OpenAI API key from the [OpenAI website](https://platform.openai.com/account/api-keys).
    """)

@cache_result(ttl=CACHE_TTL)
def extract_text_from_txt(file) -> str:
    """Extract text from a TXT file."""
    return file.read().decode("utf-8")

@cache_result(ttl=CACHE_TTL)
def extract_text_from_docx(file) -> str:
    """Extract text from a DOCX file."""
    document = Document(file)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])

def count_tokens(text: str) -> int:
    """Count tokens in a text using tiktoken or a fallback estimation."""
    try:
        # Using cl100k_base for GPT-4, GPT-3.5-turbo, etc.
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback to a simpler word-count-based estimation
        return len(text.split()) // 4  # Roughly 4 chars per token, or 1 token per 4 characters/word

@cache_result(ttl=CACHE_TTL)
def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file using multiple fallback methods with OCR support."""
    file_content = file.read()
    
    def try_pymupdf(file_content: bytes, password: str = None) -> Optional[str]:
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            if doc.is_encrypted:
                if password:
                    doc.authenticate(password)
                else:
                    raise FileProcessingError("PDF is password protected. Please provide a password.")
            
            text = ""
            total_pages = len(doc)
            
            for i, page in enumerate(doc):
                # Try normal text extraction first
                page_text = page.get_text()
                
                # If no text found, try OCR
                if not page_text.strip():
                    try:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        page_text = pytesseract.image_to_string(img)
                    except Exception as e:
                        logger.warning(f"OCR failed for page {i+1}: {e}")
                
                text += page_text + "\n"
                
                # Report progress
                progress = (i + 1) / total_pages
                st.progress(progress, text=f"Processing page {i+1} of {total_pages}")
            
            return text if text.strip() else None
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return None

    def try_pdfminer(file_content: bytes, password: str = None) -> Optional[str]:
        try:
            text = pdf_extract_text(
                BytesIO(file_content),
                password=password,
                codec='utf-8',
                laparams=LAParams(
                    line_margin=0.5,
                    word_margin=0.1,
                    char_margin=2.0,
                    boxes_flow=0.5,
                    detect_vertical=True
                )
            )
            return text if text.strip() else None
        except Exception as e:
            logger.error(f"pdfminer.six extraction failed: {e}")
            return None

    def clean_text(text: str) -> str:
        """Clean and format extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        # Fix common OCR errors
        text = text.replace('|', 'I')  # Common OCR error
        text = text.replace('l', 'I')  # Common OCR error
        # Normalize line endings
        text = text.replace('\r\n', '\n')
        # Remove empty lines
        text = '\n'.join(line for line in text.splitlines() if line.strip())
        return text.strip()

    # Check if file is password protected
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        is_encrypted = doc.is_encrypted
        doc.close()
    except Exception:
        is_encrypted = False

    password = None
    if is_encrypted:
        password = st.text_input("This PDF is password protected. Please enter the password:", type="password")
        if not password:
            raise FileProcessingError("Password is required for this PDF.")

    # Try PyMuPDF first (more robust)
    text = try_pymupdf(file_content, password)
    if text:
        logger.info("Text extracted using PyMuPDF.")
        return clean_text(text)

    # Fallback to pdfminer.six
    text = try_pdfminer(file_content, password)
    if text:
        logger.info("Text extracted using pdfminer.six.")
        return clean_text(text)

    raise FileProcessingError("Failed to extract text from PDF using all available methods. The PDF might be scanned or contain only images.")

@cache_result(ttl=CACHE_TTL)
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split text into overlapping chunks while preserving context, based on tokens."""
    if not text.strip():
        return []
    
    sentences = text.split('. ') # Simple sentence splitting
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence + ". ") # Add back period for token counting
        
        # If adding the next sentence exceeds chunk size, finalize current chunk
        if current_chunk_tokens + sentence_tokens > chunk_size and current_chunk_sentences:
            chunk_text = ". ".join(current_chunk_sentences) + "."
            chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_sentences = current_chunk_sentences[-(overlap // 2):] if len(current_chunk_sentences) > overlap // 2 else []
            current_chunk_sentences = overlap_sentences + [sentence]
            current_chunk_tokens = sum(count_tokens(s + ". ") for s in current_chunk_sentences)
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
            
    # Add the last chunk if any sentences remain
    if current_chunk_sentences:
        chunk_text = ". ".join(current_chunk_sentences) + "."
        chunks.append(chunk_text)
    
    # Ensure at least one chunk exists for very short texts
    if not chunks and text.strip():
        chunks.append(text.strip())

    logger.info(f"Text chunked into {len(chunks)} chunks, total tokens: {sum(count_tokens(c) for c in chunks)}")
    return chunks

async def stream_file(file, chunk_size: int = CHUNK_READ_SIZE) -> AsyncGenerator[bytes, None]:
    """Asynchronously stream file content in chunks."""
    while True:
        chunk = await asyncio.to_thread(file.read, chunk_size)
        if not chunk:
            break
        yield chunk

async def process_large_file(file) -> str:
    """Process a large file by streaming and extracting text."""
    full_text_buffer = io.StringIO()
    async for chunk in stream_file(file):
        full_text_buffer.write(chunk.decode('utf-8')) # Assuming utf-8 for text files
    return full_text_buffer.getvalue()

@handle_errors
async def get_ai_response_async(messages: List[Dict], stream: bool = True) -> AsyncGenerator[str, None]:
    """Get streaming AI response from OpenAI API."""
    api_key = get_api_key()
    if not api_key:
        raise AIResponseError("OpenAI API key not found. Please configure your API key.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4-turbo-preview",
                    "messages": messages,
                    "stream": stream,
                    "temperature": 0.7
                },
                timeout=30
            ) as response:
                if response.status == 401:
                    raise AIResponseError("Unauthorized: Invalid OpenAI API key. Please check your configuration.")
                elif response.status == 429:
                    raise AIResponseError("Rate limit exceeded: Too many requests to OpenAI API. Please try again later.")
                elif response.status != 200:
                    raise AIResponseError(f"OpenAI API request failed: {response.status}")

                if stream:
                    async for line in response.content:
                        if line:
                            try:
                                line = line.decode('utf-8').strip()
                                if line.startswith('data: '):
                                    data = line[6:]  # Remove 'data: ' prefix
                                    if data == '[DONE]':
                                        break
                                    try:
                                        chunk = json.loads(data)
                                        if 'choices' in chunk and len(chunk['choices']) > 0:
                                            delta = chunk['choices'][0].get('delta', {})
                                            if 'content' in delta:
                                                yield delta['content']
                                    except json.JSONDecodeError:
                                        continue
                            except Exception as e:
                                logger.error(f"Error processing stream line: {e}")
                                continue
                else:
                    result = await response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        yield result['choices'][0]['message']['content']

        except asyncio.TimeoutError:
            raise AIResponseError("OpenAI API request timed out.", traceback.format_exc())
        except Exception as e:
            raise AIResponseError(f"Error during OpenAI API request: {str(e)}", traceback.format_exc())

def get_ai_response(messages: List[Dict], stream: bool = True):
    """Synchronous wrapper for asynchronous AI response retrieval."""
    logger.info(f"API Key status before async call: {bool(get_api_key())}")
    for chunk in asyncio.run(get_ai_response_async(messages, stream=stream)):
        yield chunk

@handle_errors
def process_transcript(uploaded_file) -> Optional[List[str]]:
    """Process an uploaded transcript file."""
    if uploaded_file is None:
        st.warning("No file uploaded.")
        return None

    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type not in VALID_TRANSCRIPT_TYPES:
        raise ValidationError(f"Unsupported file type: {file_type}. Please upload a .txt, .docx, or .pdf file.")

    start_time = time.time()
    full_text = ""
    
    with st.spinner(f"Processing {uploaded_file.name}..."):
        if file_type == "txt":
            full_text = extract_text_from_txt(uploaded_file)
        elif file_type == "docx":
            full_text = extract_text_from_docx(uploaded_file)
        elif file_type == "pdf":
            full_text = extract_text_from_pdf(uploaded_file)
        
    if not full_text.strip():
        raise FileProcessingError("Extracted text is empty. The file might be unreadable or empty.")

    end_time = time.time()
    logger.info(f"File processed in {end_time - start_time:.2f} seconds")

    # Store full text in session state for analysis
    st.session_state.transcript_text = full_text
    st.success("File uploaded and text extracted successfully!")

    return [full_text] # Return as a list of strings, e.g., for simple display

def highlight_text(text, theme, color):
    """Highlight text with a given color and theme tag."""
    return f'<span style="background-color:{color}; padding: 2px 5px; border-radius: 3px;">{text} <sup style="font-size: 0.7em; opacity: 0.8;">🔹[{theme}]</sup></span>'

def load_knowledge_base() -> Dict[str, Dict[str, Any]]:
    """Load knowledge base from session state or initialize empty."""
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = {}
    return st.session_state.knowledge_base

def save_knowledge_base(kb: Dict[str, Dict[str, Any]]):
    """Save knowledge base to session state."""
    st.session_state.knowledge_base = kb

def create_theme_legend():
    st.markdown("### Theme Legend")
    legend_html = """
    <div style="display: flex; flex-wrap: wrap; gap: 10px;">
    """
    for theme, color in THEME_COLORS.items():
        legend_html += f"""
        <div style="display: flex; align-items: center; font-size: 0.9em;">
            <span style="display: inline-block; width: 15px; height: 15px; background-color: {color}; border-radius: 3px; margin-right: 5px;"></span>
            {theme}
        </div>
        """
    legend_html += "</div>"
    st.sidebar.markdown(legend_html, unsafe_allow_html=True)

def display_coded_segment(segment: Dict[str, Any]):
    """Display a single coded segment."""
    text = segment.get("text", "")
    code = segment.get("code", "No Code")
    theme = segment.get("theme", "No Theme")
    notes = segment.get("notes", "")
    
    color = THEME_COLORS.get(theme, "#CCCCCC") # Default to grey if theme not found

    st.markdown(highlight_text(text, theme, color), unsafe_allow_html=True)
    if notes:
        st.markdown(f"*Notes:* {notes}")
    st.markdown("---")

def create_theme_distribution_chart(themes: List[str]):
    """Create a Plotly pie chart for theme distribution."""
    theme_counts = pd.Series(themes).value_counts().reset_index()
    theme_counts.columns = ['Theme', 'Count']
    
    fig = px.pie(theme_counts, 
                 values='Count', 
                 names='Theme', 
                 title='Theme Distribution',
                 color='Theme',
                 color_discrete_map=THEME_COLORS)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("Qualinsight AI")
        st.markdown("---")
        
        # Display theme legend
        create_theme_legend()
        
        # Display analysis duration if available
        if st.session_state.analysis_duration is not None:
            st.write(f"Last Analysis Duration: {st.session_state.analysis_duration:.2f} seconds")
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Upload", "Analysis", "Results", "Export"],
            icons=["upload", "search", "list", "download"],
            default_index=0
        )
    
    # Main content area
    if selected == "Upload":
        handle_upload_tab()
    elif selected == "Analysis":
        handle_analysis_tab()
    elif selected == "Results":
        handle_results_tab()
    elif selected == "Export":
        handle_export_tab()

@handle_errors
def handle_upload_tab():
    """Handle the file upload tab content."""
    st.header("Upload Transcript")
    st.write("Upload your transcript file (TXT, DOCX, or PDF) to start the analysis.")

    # Input for OpenAI API Key (new)
    current_api_key_in_session = st.session_state.get("openai_api_key_input", "")
    initial_api_key_from_sources = get_api_key()

    # If an API key is already valid from other sources, use that as the initial value
    # Otherwise, use what's in session state (if user typed something previously)
    # This ensures user's typed input takes precedence if it exists
    display_value = current_api_key_in_session if current_api_key_in_session else (initial_api_key_from_sources or "")

    st.session_state.openai_api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        value=display_value,
        help="Enter your OpenAI API key. It will not be stored permanently."
    )

    # After user input, check if a valid API key is now available
    valid_api_key_exists = get_api_key() # This will now prioritize session state input

    if not valid_api_key_exists:
        show_api_key_guidance()
    else:
        st.success("OpenAI API key configured successfully!")

    uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx", "pdf"])
    if uploaded_file is not None:
        process_transcript(uploaded_file)

    # Research Question input
    st.session_state.research_question = st.text_area(
        "Enter your Research Question",
        value=st.session_state.get("research_question", ""),
        help="This question will guide the AI's analysis of your transcript."
    )

@handle_errors
def handle_analysis_tab():
    """Handle the analysis tab with improved async processing."""
    logger.info("Entering handle_analysis_tab.")
    if not st.session_state.transcript_text:
        st.warning("Please upload a transcript first.")
        logger.info("No transcript text, returning from handle_analysis_tab.")
        return
    
    st.header("Analysis Progress")
    
    # Analysis controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.research_question = st.text_area(
            "Research Question",
            value=st.session_state.get("research_question", ""),
            help="Enter your research question to guide the analysis"
        )
    
    with col2:
        if st.button("Start Analysis", type="primary"):
            logger.info("Start Analysis button clicked.")
            # Clear previous results and set initial state for new analysis
            st.session_state.analysis_results = None
            st.session_state.analysis_complete = False

            task_id = start_analysis_task(
                st.session_state.transcript_text,
                st.session_state.research_question,
                st.session_state.analysis_mode,
                st.session_state.framework
            )
            st.session_state.current_task = task_id
            logger.info(f"Analysis task started, current_task set to {task_id}")
            st.rerun() # Force rerun to show progress immediately
    
    # Show progress for current task
    if "current_task" in st.session_state:
        task_id = st.session_state.current_task
        progress = st.session_state.get(f"progress_{task_id}", 0.0)
        status = st.session_state.get(f"status_{task_id}", "Starting analysis...")
        analysis_complete = st.session_state.get("analysis_complete", False)
        
        # Retrieve the progress queue
        progress_queue = task_manager.get_progress_queue(task_id)
        if progress_queue:
            # Check if there are any new messages in the queue
            queue_had_messages = False
            while not progress_queue.empty():
                queue_had_messages = True
                try:
                    message_type, *data = progress_queue.get_nowait()
                    if message_type == "progress":
                        progress_val, status_msg = data
                        st.session_state[f"progress_{task_id}"] = progress_val
                        st.session_state[f"status_{task_id}"] = status_msg
                        progress = progress_val # Update local variable for current render cycle
                        status = status_msg
                    elif message_type == "complete_data":
                        analysis_results = data[0]
                        st.session_state.analysis_results = analysis_results
                        st.session_state.analysis_complete = True
                        st.session_state[f"status_{task_id}"] = "Analysis complete!"
                        analysis_complete = True # Update local variable
                    elif message_type == "error":
                        error_msg = data[0]
                        st.session_state[f"status_{task_id}"] = f"Error: {error_msg}"
                        st.session_state.analysis_complete = True
                        analysis_complete = True # Update local variable
                except Exception as e:
                    logger.error(f"Error processing queue message for task {task_id}: {e}", exc_info=True)
                    # Don't re-raise, allow the app to continue

            # Update the progress bar and status text again after processing queue
            progress_bar = st.progress(progress)
            status_text = st.empty()

            progress_bar.progress(progress)
            status_text.text(status)

            if analysis_complete:
                logger.info(f"Task {task_id} detected as complete in UI. Results: {st.session_state.get('analysis_results') is not None}")
                progress_bar.empty() # Clear progress bar
                status_text.empty() # Clear status text
                
                if st.session_state.get("analysis_results"):
                    status_box.update(label="Analysis complete!", state="complete", expanded=False)
                    display_analysis_results(st.session_state.analysis_results)
                elif status.startswith("Error"):
                    status_box.update(label=f"Analysis failed: {status}", state="error", expanded=True)
                else:
                    status_box.update(label="Analysis completed with no results or unexpected state.", state="complete", expanded=False)

                # Clean up task and session state for the next run
                task_manager.cleanup(task_id)
                st.session_state.active_tasks.discard(task_id)
                if "current_task" in st.session_state:
                    del st.session_state.current_task
                st.session_state.analysis_complete = False # Reset for next run
                st.session_state.analysis_results = None   # Reset for new analysis
                logger.info(f"Task {task_id} UI elements cleaned up.")

            elif not analysis_complete:
                # If the task is not complete and the queue is empty, wait a bit and rerun
                # This prevents excessive reruns when no new data is available
                if not queue_had_messages and not task_manager.is_complete(task_id):
                    time.sleep(0.5) # Increased delay to prevent excessive reruns
                
                logger.info(f"Task {task_id} not yet complete. Forcing rerun.")
                st.rerun()

@handle_errors
def handle_results_tab():
    """Handle the results tab content."""
    st.header("Analysis Results")
    
    if "analysis_results" not in st.session_state or not st.session_state.analysis_results:
        st.info("No analysis results available. Please run an analysis first.")
        return

    analysis_results = st.session_state.analysis_results

    if not isinstance(analysis_results, dict) or "themes" not in analysis_results or "coded_segments" not in analysis_results:
        st.error("Invalid analysis results format.")
        return

    themes = analysis_results.get("themes", [])
    coded_segments = analysis_results.get("coded_segments", [])

    if not themes and not coded_segments:
        st.info("Analysis completed, but no themes or coded segments were identified.")
        return

    # Display themes
    st.subheader("Identified Themes")
    if themes:
        for i, theme in enumerate(themes):
            st.markdown(f"**{i+1}. {theme}**")
    else:
        st.info("No themes identified.")

    # Display coded segments
    st.subheader("Coded Segments")
    if coded_segments:
        # Filter and display by theme if selected
        all_themes = sorted(list(set(s.get("theme", "No Theme") for s in coded_segments)))
        selected_theme = st.selectbox("Filter by Theme", ["All Themes"] + all_themes)

        if selected_theme == "All Themes":
            segments_to_display = coded_segments
        else:
            segments_to_display = [s for s in coded_segments if s.get("theme") == selected_theme]
        
        if segments_to_display:
            for segment in segments_to_display:
                display_coded_segment(segment)
        else:
            st.info("No coded segments for the selected theme.")
    else:
        st.info("No coded segments available.")

    # Show Theme Distribution Chart
    if themes and coded_segments:
        st.subheader("Theme Distribution")
        segment_themes = [s.get("theme", "No Theme") for s in coded_segments]
        if segment_themes:
            fig = create_theme_distribution_chart(segment_themes)
            st.plotly_chart(fig, use_container_width=True)

def display_analysis_results(analysis_results: Dict):
    """Helper function to display analysis results from handle_analysis_tab."""
    st.subheader("Analysis Summary")
    st.write(analysis_results.get("summary", "No summary provided."))

    themes = analysis_results.get("themes", [])
    coded_segments = analysis_results.get("coded_segments", [])

    if themes:
        st.subheader("Identified Themes")
        st.markdown("-" + "\n-".join(themes))

    if coded_segments:
        st.subheader("First 5 Coded Segments (for quick review)")
        for segment in coded_segments[:5]: # Show first 5 for brevity
            display_coded_segment(segment)
        if len(coded_segments) > 5:
            st.info(f"See the 'Results' tab for all {len(coded_segments)} coded segments.")

@handle_errors
def handle_export_tab():
    """Handle the export tab content."""
    st.header("Export Analysis Results")

    if "analysis_results" not in st.session_state or not st.session_state.analysis_results:
        st.info("No analysis results available to export. Please run an analysis first.")
        return

    analysis_results = st.session_state.analysis_results
    coded_segments = analysis_results.get("coded_segments", [])

    if not coded_segments:
        st.warning("No coded segments to export.")
        return

    df_segments = pd.DataFrame(coded_segments)

    st.subheader("Export Coded Segments")

    # CSV Export
    csv_data = df_segments.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Coded Segments as CSV",
        data=csv_data,
        file_name="coded_segments.csv",
        mime="text/csv",
    )

    # PDF Export
    if st.button("Download Coded Segments as PDF"):
        try:
            pdf_buffer = BytesIO()
            doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
            styles = getSampleStyleSheet()
            
            flowables = []
            flowables.append(Paragraph("Qualinsight AI - Coded Segments Report", styles['h1']))
            flowables.append(Spacer(1, 0.2 * inch))

            # Add summary if available
            summary = analysis_results.get("summary")
            if summary:
                flowables.append(Paragraph("Analysis Summary:", styles['h2']))
                flowables.append(Paragraph(summary, styles['Normal']))
                flowables.append(Spacer(1, 0.2 * inch))
            
            flowables.append(Paragraph("Coded Segments:", styles['h2']))

            data = [["Theme", "Code", "Segment Text", "Notes"]]
            for _, row in df_segments.iterrows():
                data.append([row['theme'], row['code'], row['text'], row['notes']])

            # Create table with styles for better readability
            table_style = [
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black)
            ]
            table = Table(data, colWidths=[1.5*inch, 1.5*inch, 3*inch, 1.5*inch])
            table.setStyle(TableStyle(table_style))
            flowables.append(table)
            
            doc.build(flowables)
            pdf_buffer.seek(0)
            st.download_button(
                label="Click to Download PDF",
                data=pdf_buffer.getvalue(),
                file_name="coded_segments.pdf",
                mime="application/pdf",
            )
            st.success("PDF generated successfully!")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            st.exception(e)

def merge_coded_chunks(chunks: List[Dict]) -> List[Dict]:
    """Merges overlapping or adjacent coded chunks into larger segments if they share the same theme and code."""
    if not chunks:
        return []

    # Sort chunks by their original position/index if available, or by text for consistency
    # Assuming chunks might have an 'original_index' or are processed in order
    # For simplicity, if not, we'll sort by text content or a placeholder
    # In a real app, you'd track original segment IDs/ranges for robust merging
    sorted_chunks = sorted(chunks, key=lambda x: x.get("text", "")) # Simple sort for now

    merged_segments = []
    current_segment = None

    for chunk in sorted_chunks:
        if current_segment is None:
            current_segment = {
                "text": chunk.get("text", ""),
                "code": chunk.get("code", ""),
                "theme": chunk.get("theme", ""),
                "notes": chunk.get("notes", ""),
                "original_chunks": [chunk] # Keep track of original chunks
            }
        else:
            # Check for same theme and code, and if chunks are 'adjacent' (conceptual adjacency here)
            # A more sophisticated check would involve original text offsets
            if (chunk.get("theme") == current_segment["theme"] and
                chunk.get("code") == current_segment["code"]):
                
                # Simple merge: concatenate text and notes
                current_segment["text"] += " " + chunk.get("text", "")
                if chunk.get("notes") and chunk.get("notes") not in current_segment["notes"]:
                    current_segment["notes"] += "; " + chunk.get("notes", "")
                current_segment["original_chunks"].append(chunk)
            else:
                merged_segments.append(current_segment)
                current_segment = {
                    "text": chunk.get("text", ""),
                    "code": chunk.get("code", ""),
                    "theme": chunk.get("theme", ""),
                    "notes": chunk.get("notes", ""),
                    "original_chunks": [chunk]
                }
    
    if current_segment:
        merged_segments.append(current_segment)

    # Optionally, you might want to clean up merged_segments to remove original_chunks
    for segment in merged_segments:
        if "original_chunks" in segment:
            del segment["original_chunks"]

    return merged_segments

def calculate_intercoder_reliability(coders_data: List[pd.DataFrame]) -> Dict[str, float]:
    """Calculate inter-coder reliability (Cohen's Kappa) for pairs of coders."""
    if len(coders_data) < 2:
        return {"error": "At least two coder dataframes are required for inter-coder reliability calculation.", "kappa_scores": {}}

    kappa_scores = {}

    # Extract all unique segments across all coders (simplified: using text as identifier)
    all_segments_text = set()
    for df in coders_data:
        all_segments_text.update(df['text'].unique())
    
    # Create a unified DataFrame for comparison
    # This is a simplified approach. A robust solution needs careful handling of segment boundaries and overlaps.
    unified_data = pd.DataFrame(list(all_segments_text), columns=['text'])

    for i, j in combinations(range(len(coders_data)), 2):
        coder1_name = f"Coder_{i+1}"
        coder2_name = f"Coder_{j+1}"

        df1 = coders_data[i].set_index('text')
        df2 = coders_data[j].set_index('text')

        # Merge their codings
        merged_df = unified_data.merge(df1[['theme']], left_on='text', right_index=True, how='left', suffixes= ('' , '_coder1'))
        merged_df = merged_df.merge(df2[['theme']], left_on='text', right_index=True, how='left', suffixes= ('_coder1' , '_coder2'))

        # Fill NaNs with a placeholder like 'UNCODED' for kappa calculation
        merged_df['theme_coder1'] = merged_df['theme_coder1'].fillna('UNCODED')
        merged_df['theme_coder2'] = merged_df['theme_coder2'].fillna('UNCODED')

        # Calculate Cohen's Kappa
        try:
            kappa = cohen_kappa_score(merged_df['theme_coder1'], merged_df['theme_coder2'])
            kappa_scores[f"{coder1_name} vs {coder2_name}"] = kappa
        except Exception as e:
            logger.error(f"Error calculating kappa for {coder1_name} vs {coder2_name}: {e}")
            kappa_scores[f"{coder1_name} vs {coder2_name}"] = float('nan') # Not a Number

    return {"kappa_scores": kappa_scores}

def load_framework(file) -> Dict:
    """Load a framework from a JSON file."""
    try:
        framework_content = file.read().decode("utf-8")
        framework = json.loads(framework_content)
        # Basic validation for framework structure
        if "themes" not in framework or not isinstance(framework["themes"], list):
            raise ValidationError("Framework JSON must contain a 'themes' list.")
        for theme in framework["themes"]:
            if "name" not in theme or "description" not in theme or not isinstance(theme["name"], str):
                raise ValidationError("Each theme in framework must have a 'name' (string) and 'description'.")
        return framework
    except json.JSONDecodeError as e:
        raise FileProcessingError(f"Invalid JSON in framework file: {e}")
    except Exception as e:
        raise FileProcessingError(f"Error loading framework: {e}")

def ai_generate_codes(text: str, mode: str, rq: str, framework: Optional[str] = None) -> List[dict]:
    """Simulate AI generating codes for a text segment."""
    logger.info(f"ai_generate_codes called with mode: {mode}, text length: {len(text)}")
    # This is a placeholder. Replace with actual AI call.
    # For demonstration, generate random themes and codes
    possible_themes = list(THEME_COLORS.keys())
    
    # Dummy AI response structure
    if mode == "Thematic Analysis":
        response_structure = {
            "coded_segments": [
                {
                    "text": text,
                    "code": f"Code {random.randint(1, 5)}",
                    "theme": random.choice(possible_themes),
                    "notes": "AI generated note."
                }
            ]
        }
    elif mode == "Grounded Theory":
         response_structure = {
            "coded_segments": [
                {
                    "text": text,
                    "code": f"InVivoCode {random.randint(1, 3)}",
                    "theme": "Emergent Theme",
                    "notes": "Grounded theory initial coding."
                }
            ]
        }
    elif mode == "Framework Analysis":
        if not framework or "themes" not in framework:
            raise ValidationError("Framework is required for Framework Analysis.")
        
        framework_themes = [t["name"] for t in framework["themes"]]
        response_structure = {
            "coded_segments": [
                {
                    "text": text,
                    "code": f"FrameworkCode {random.randint(1, 3)}",
                    "theme": random.choice(framework_themes), # Select from framework themes
                    "notes": "Framework analysis applied."
                }
            ]
        }
    else:
        raise ValidationError(f"Unknown analysis mode: {mode}")

    # Simulate API call delay
    time.sleep(0.5) 

    # Validate the dummy response against a simplified schema for safety
    try:
        validated_response = validate_ai_response(response_structure, {"coded_segments": list})
        # Further validate each segment within coded_segments if needed
        logger.info(f"ai_generate_codes completed for mode: {mode}")
        return validated_response["coded_segments"]
    except AIResponseError as e:
        logger.error(f"AI response validation failed: {e.message}")
        raise # Re-raise for error handling decorator

@cache_result(ttl=CACHE_TTL)
def analyze_transcript(transcript_text: str, research_question: str, analysis_mode: str, framework: Optional[Dict] = None) -> Dict:
    """Analyze the transcript using the selected AI model and mode."""
    logger.info(f"Starting transcript analysis for mode: {analysis_mode}, RQ: {research_question}")
    
    chunks = chunk_text(transcript_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    all_coded_segments = []
    all_themes = set()
    
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        # Simulate AI processing each chunk
        # In a real scenario, this would be an API call to a coding model
        coded_segments_for_chunk = ai_generate_codes(chunk, analysis_mode, research_question, framework)
        all_coded_segments.extend(coded_segments_for_chunk)
        for segment in coded_segments_for_chunk:
            if "theme" in segment:
                all_themes.add(segment["theme"])
        
        # Update progress and status (this is where the UI update would happen)
        progress = (i + 1) / total_chunks
        status = f"Analyzing chunk {i+1}/{total_chunks}..."
        logger.info(f"Progress: {progress:.2f}, Status: {status}")

    merged_segments = merge_coded_chunks(all_coded_segments)

    # Create a dummy summary
    summary = f"Analysis complete for research question: '{research_question}'. Identified {len(all_themes)} themes and {len(merged_segments)} coded segments."

    return {
        "summary": summary,
        "themes": sorted(list(all_themes)),
        "coded_segments": merged_segments
    }

async def process_chunk_async(chunk: str, research_question: str, analysis_mode: str, framework: Optional[Dict] = None) -> Dict:
    """Asynchronously process a single chunk using AI."""
    logger.info(f"process_chunk_async called for chunk length: {len(chunk)}")
    # Simulate AI API call, replace with actual API interaction
    # This part should ideally use an aiohttp call to OpenRouter or similar
    await asyncio.sleep(0.1) # Simulate async work

    # Call the synchronous ai_generate_codes in a thread pool executor
    # This is crucial because ai_generate_codes might be blocking (e.g., if it makes synchronous HTTP requests)
    loop = asyncio.get_event_loop()
    coded_segments = await loop.run_in_executor(
        THREAD_POOL,
        lambda: ai_generate_codes(chunk, analysis_mode, research_question, framework)
    )
    logger.info(f"process_chunk_async completed for chunk length: {len(chunk)}")
    return {"coded_segments": coded_segments}

async def process_transcript_async(transcript_text: str, research_question: str, analysis_mode: str, framework: Optional[Dict] = None) -> AsyncGenerator[Tuple[float, Dict], None]:
    """Process the transcript asynchronously with progress updates."""
    logger.info(f"process_transcript_async started for transcript length: {len(transcript_text)}")
    chunks = chunk_text(transcript_text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    total_chunks = len(chunks)
    all_coded_segments = []
    all_themes = set()

    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{total_chunks} in process_transcript_async.")
        try:
            chunk_result = await process_chunk_async(chunk, research_question, analysis_mode, framework)
            coded_segments_for_chunk = chunk_result.get("coded_segments", [])
            all_coded_segments.extend(coded_segments_for_chunk)
            for segment in coded_segments_for_chunk:
                if "theme" in segment:
                    all_themes.add(segment["theme"])
            
            progress = (i + 1) / total_chunks
            status = f"Analyzing chunk {i+1}/{total_chunks}..."
            logger.info(f"Yielding progress: {progress:.2f}, status: {status}")
            yield progress, {"status": status, "type": "progress"}
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}", exc_info=True)
            yield -1.0, {"status": f"Error processing chunk {i+1}: {e}", "type": "error"}
            return # Stop processing on error

    merged_segments = merge_coded_chunks(all_coded_segments)
    summary = f"Analysis complete for research question: '{research_question}'. Identified {len(all_themes)} themes and {len(merged_segments)} coded segments."

    final_results = {
        "summary": summary,
        "themes": sorted(list(all_themes)),
        "coded_segments": merged_segments
    }
    logger.info(f"process_transcript_async completed. Final results: {final_results["summary"]}")
    yield 1.0, {"status": "Analysis complete!", "type": "complete", "results": final_results}

def start_analysis_task(transcript_text: str, research_question: str, analysis_mode: str, framework: Optional[Dict] = None) -> str:
    """Starts the analysis as an asynchronous task and returns its ID."""
    task_id = str(uuid.uuid4()) # Unique ID for the task
    
    # Create a queue specifically for this task's progress updates
    progress_queue = Queue()

    # Pass the progress_queue to the async analysis function
    async def analysis_coroutine():
        async for progress, data_dict in process_transcript_async(transcript_text, research_question, analysis_mode, framework):
            if data_dict["type"] == "progress":
                progress_queue.put(("progress", progress, data_dict["status"]))
            elif data_dict["type"] == "complete":
                progress_queue.put(("complete_data", data_dict["results"]))
                break
            elif data_dict["type"] == "error":
                progress_queue.put(("error", data_dict["status"]))
                break
        # Ensure final state is always reported to the main thread
        # (This is handled by 'complete_data' or 'error' messages)

    task_manager.start_task(task_id, analysis_coroutine(), progress_queue)
    
    # Initialize session state for this specific task's progress
    st.session_state[f"progress_{task_id}"] = 0.0
    st.session_state[f"status_{task_id}"] = "Starting analysis..."
    
    st.session_state.analysis_start_time = time.time()
    st.session_state.analysis_duration = None # Reset duration for new analysis

    return task_id

# Run the main application function
if __name__ == "__main__":
    main()