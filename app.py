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
from functools import lru_cache, wraps
import hashlib
import logging
from openai import OpenAI
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
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-4o" # Default model

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
    """Validate the format of the provided API key."""
    # Basic validation: check if it's a non-empty string and starts with 'sk-'
    return isinstance(api_key, str) and api_key.strip() != "" # and api_key.startswith("sk-") # OpenRouter keys don't necessarily start with sk-"

def get_api_key() -> Optional[str]:
    """Retrieve the API key from Streamlit secrets or environment variables."""
    api_key = None
    try:
        # Attempt to retrieve from Streamlit secrets
        if "OPENROUTER_API_KEY" in st.secrets:
            api_key = st.secrets["OPENROUTER_API_KEY"]
            logger.info("API key found in Streamlit secrets.")
        elif "OPENROUTER_API_KEY" in os.environ:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            logger.info("API key found in environment variables.")
        else:
            logger.warning("OPENROUTER_API_KEY not found in Streamlit secrets or environment variables.")
            return None
        
        if not validate_api_key(api_key):
            logger.error("Retrieved API key is invalid.")
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving API key: {e}")
        return None
    
    return api_key

def show_api_key_guidance():
    st.warning("OpenRouter API Key Not Configured")
    st.markdown("""
    To use this application, you need to provide your OpenRouter API key. 
    Please configure it using one of the following methods:

    1.  **Streamlit Secrets (Recommended for Deployment):**
        Create a `.streamlit/secrets.toml` file in your project root with the following content:
        ```toml
        OPENROUTER_API_KEY="your_openrouter_api_key_here"
        ```
        **Note:** Do not commit `secrets.toml` to public GitHub repositories.

    2.  **Environment Variable (Recommended for Local Development):**
        Set the `OPENROUTER_API_KEY` environment variable before running the app:
        ```bash
        export OPENROUTER_API_KEY="your_openrouter_api_key_here"
        streamlit run app.py
        ```
        On Windows (Command Prompt):
        ```cmd
        set OPENROUTER_API_KEY="your_openrouter_api_key_here"
        streamlit run app.py
        ```
    You can obtain your OpenRouter API key from the [OpenRouter website](https://openrouter.ai/keys).
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
    """Extract text from a PDF file using multiple fallback methods."""
    file_content = file.read()
    
    def try_pymupdf(file_content: bytes) -> Optional[str]:
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            return text if text.strip() else None
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return None

    def try_pdfminer(file_content: bytes) -> Optional[str]:
        try:
            text = pdf_extract_text(BytesIO(file_content))
            return text if text.strip() else None
        except Exception as e:
            logger.error(f"pdfminer.six extraction failed: {e}")
            return None

    # Try PyMuPDF first (more robust)
    text = try_pymupdf(file_content)
    if text:
        logger.info("Text extracted using PyMuPDF.")
        return text

    # Fallback to pdfminer.six
    text = try_pdfminer(file_content)
    if text:
        logger.info("Text extracted using pdfminer.six.")
        return text

    raise FileProcessingError("Failed to extract text from PDF using all available methods.")

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
    """Get streaming AI response from OpenRouter API."""
    api_key = get_api_key()
    if not api_key:
        raise ValidationError("OpenRouter API key is missing or invalid. Please configure it.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501", # Replace with your app's URL
        "X-Title": "Qualinsight AI",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
    }

    timeout_seconds = 300  # 5 minutes for API response

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as session:
            async with session.post(API_URL, headers=headers, json=payload) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
                
                async for chunk in response.content.iter_any():
                    try:
                        # OpenRouter sends SSEs (Server-Sent Events) which start with 'data: '
                        # We need to parse each line and extract the JSON part.
                        for line in chunk.decode('utf-8').splitlines():
                            if line.startswith("data:"):
                                json_data = line[len("data:"):].strip()
                                if json_data == "[DONE]":
                                    break
                                
                                data = json.loads(json_data)
                                if "choices" in data and data["choices"] and "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                                    content = data["choices"][0]["delta"]["content"]
                                    if content:
                                        yield content
                    except json.JSONDecodeError:
                        logger.warning(f"JSONDecodeError: Could not decode line: {line.strip()}")
                        continue

    except aiohttp.ClientError as e:
        error_message = f"OpenRouter API request failed: {e}"
        if isinstance(e, aiohttp.ClientResponseError):
            if e.status == 401:
                error_message = "Unauthorized: Invalid OpenRouter API key. Please check your configuration."
            elif e.status == 429:
                error_message = "Rate limit exceeded: Too many requests to OpenRouter API. Please try again later."
            else:
                try:
                    response_text = await response.text()
                    error_message += f"\nResponse: {response_text}"
                except Exception:
                    pass
        raise AIResponseError(error_message, traceback.format_exc())
    except asyncio.TimeoutError:
        raise AIResponseError("OpenRouter API request timed out.", traceback.format_exc())
    except Exception as e:
        raise AIResponseError(f"An unexpected error occurred during API call: {e}", traceback.format_exc())

def get_ai_response(messages: List[Dict], stream: bool = True):
    """Synchronous wrapper for AI response (for non-streaming or compatibility)."""
    # This function is now a placeholder as primary AI calls are async.
    # For synchronous use, you'd run the async generator to completion.
    full_response_content = ""
    try:
        for chunk in asyncio.run(get_ai_response_async(messages, stream=stream)):
            full_response_content += chunk
        return full_response_content
    except Exception as e:
        st.error(f"Error in get_ai_response: {e}")
        return ""

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
    """Handle the upload tab content."""
    st.header("Upload Transcript")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your transcript (TXT, DOCX, or PDF)",
        type=VALID_TRANSCRIPT_TYPES,
        help=f"Max file size: {MAX_FILE_SIZE / (1024 * 1024):.0f} MB"
    )

    if uploaded_file:
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
            while not progress_queue.empty():
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

        logger.info(f"Monitoring task {task_id}. Progress: {progress:.1%}, Status: {status}, Complete: {analysis_complete}")

        with st.status("Analysis Status", expanded=True) as status_box:
            progress_bar = st.progress(progress)
            status_text = st.empty()

            progress_bar.progress(progress)
            status_text.text(status)

            if analysis_complete:
                logger.info(f"Task {task_id} detected as complete in UI. Results: {st.session_state.get("analysis_results") is not None}")
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
                st.session_state.analysis_results = None   # Reset for next run
                logger.info(f"Task {task_id} UI elements cleaned up.")

            else:
                # Rerun the app to update progress until analysis is complete
                logger.info(f"Task {task_id} not yet complete. Forcing rerun.")
                time.sleep(0.1) # Small delay to prevent excessive reruns
                st.rerun()

@handle_errors
def handle_results_tab():
    """Handle the results tab with error handling."""
    if not st.session_state.analysis_complete:
        st.warning("Please complete the analysis first.")
        return
        
    st.header("Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Coded Transcript", "Theme Analysis", "Statistics"])
    
    with tab1:
        st.subheader("Coded Transcript")
        if not st.session_state.analysis_results.get("coded_segments"):
            st.warning("No coded segments found in the analysis results.")
            return
            
        for segment in st.session_state.analysis_results["coded_segments"]:
            display_coded_segment(segment)
    
    with tab2:
        st.subheader("Theme Analysis")
        col1, col2 = st.columns([2, 1])
        with col1:
            # Theme distribution chart
            themes = [seg.get("theme") for seg in st.session_state.analysis_results.get("coded_segments", [])]
            if not themes:
                st.warning("No themes found in the analysis results.")
                return
                
            st.plotly_chart(create_theme_distribution_chart(themes), use_container_width=True)
        
        with col2:
            # Theme list with counts
            theme_counts = pd.Series(themes).value_counts()
            st.markdown("### Theme Frequency")
            for theme, count in theme_counts.items():
                st.markdown(f"- **{theme}**: {count} segments")
    
    with tab3:
        st.subheader("Analysis Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Segments", len(st.session_state.analysis_results.get("coded_segments", [])))
            st.metric("Unique Themes", len(set(themes)))
        with col2:
            st.metric("Analysis Time", f"{st.session_state.analysis_results.get('analysis_time', 0):.2f} seconds")

@handle_errors
def handle_export_tab():
    """Handle the export tab with error handling."""
    if not st.session_state.analysis_complete:
        st.warning("Please complete the analysis first.")
        return
        
    st.header("Export Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Export Options")
        export_format = st.selectbox(
            "Format",
            ["PDF", "DOCX", "CSV"],
            help="Choose the export format"
        )
        
        if st.button("Export", type="primary"):
            with st.spinner("Preparing export..."):
                try:
                    # Export logic here
                    st.success("Export completed!")
                except Exception as e:
                    raise FileProcessingError(
                        "Failed to export results",
                        f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
                    )
    
    with col2:
        st.subheader("Preview")
        st.markdown("### Export Preview")
        # Preview logic here

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
    all_segments = sorted(list(set(segment for coder_df in coders_data for segment in coder_df['segment'])))
    
    # Create a DataFrame for all segments and coders' codes
    segment_codes = {segment: [] for segment in all_segments}
    for coder_df in coders_data:
        for segment in all_segments:
            # Find the code for the current segment from the current coder
            code = coder_df[coder_df['segment'] == segment]['code'].iloc[0] if segment in coder_df['segment'].values else None
            segment_codes[segment].append(code)
            
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

def analyze_transcript(transcript_text: str, research_question: str, analysis_mode: str, framework: Optional[Dict] = None) -> Dict:
    """Analyze transcript using OpenAI API with streaming response"""
    try:
        # Create progress container
        progress_container = st.empty()
        progress_container.info("🔄 Initializing AI analysis...")
        
        # Prepare the prompt
        prompt = f"""Analyze the following transcript in the context of this research question: "{research_question}"

Transcript:
{transcript_text}

Please provide a detailed analysis including:
1. Key themes and patterns
2. Relevant quotes and examples
3. Insights and implications
4. Recommendations for further research

Format the response in clear sections with markdown formatting."""

        # Initialize OpenAI client
        client = OpenAI(api_key=st.secrets["openai"]["api_key"])
        
        # Update progress
        progress_container.info("🤖 AI is processing the content...")
        
        # Get streaming response
        stream = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        
        # Process streaming response
        full_response = ""
        response_container = st.empty()
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_container.markdown(full_response)
        
        # Update progress
        progress_container.info("✨ Analysis complete! Processing results...")
        
        # Process and structure the response
        # Split the response into sections
        sections = full_response.split('\n\n')
        themes = []
        quotes = []
        insights = []
        recommendations = []
        
        current_section = None
        for section in sections:
            if section.startswith('1.') or 'themes' in section.lower():
                current_section = 'themes'
            elif section.startswith('2.') or 'quotes' in section.lower():
                current_section = 'quotes'
            elif section.startswith('3.') or 'insights' in section.lower():
                current_section = 'insights'
            elif section.startswith('4.') or 'recommendations' in section.lower():
                current_section = 'recommendations'
            
            if current_section == 'themes':
                themes.append(section)
            elif current_section == 'quotes':
                quotes.append(section)
            elif current_section == 'insights':
                insights.append(section)
            elif current_section == 'recommendations':
                recommendations.append(section)
        
        analysis_result = {
            "themes": themes,
            "quotes": quotes,
            "insights": insights,
            "recommendations": recommendations,
            "coded_segments": [{"text": q, "code": "Auto-coded", "notes": ""} for q in quotes]
        }
        
        # Clear progress message
        progress_container.empty()
        
        return analysis_result
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return {}

async def process_chunk_async(chunk: str, research_question: str, analysis_mode: str, framework: Optional[Dict] = None) -> Dict:
    """Process a single chunk asynchronously."""
    try:
        return await analyze_transcript_async(chunk, research_question, analysis_mode, framework)
    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        raise

async def process_transcript_async(transcript_text: str, research_question: str, analysis_mode: str, framework: Optional[Dict] = None) -> AsyncGenerator[Tuple[float, Dict], None]:
    """Process transcript asynchronously with progress tracking."""
    # Chunk the text
    chunks = chunk_text(transcript_text)
    total_chunks = len(chunks)
    
    # Process chunks concurrently
    tasks = []
    for i, chunk in enumerate(chunks):
        task = asyncio.create_task(
            process_chunk_async(chunk, research_question, analysis_mode, framework)
        )
        tasks.append((i, task))
    
    # Collect results as they complete
    results = []
    completed_chunks = 0
    
    for i, task in tasks:
        try:
            result = await task
            results.append(result)
            completed_chunks += 1
            
            # Yield progress for the main thread to pick up
            progress = completed_chunks / total_chunks
            yield progress, result
        except Exception as e:
            logger.error(f"Error processing chunk {i}: {str(e)}")
            raise
    
    # Yield final merged result (progress 1.0 is handled by the last yield in the loop)
    final_result = merge_coded_chunks(results)
    yield 1.0, final_result

def start_analysis_task(transcript_text: str, research_question: str, analysis_mode: str, framework: Optional[Dict] = None) -> str:
    """Start an asynchronous analysis task."""
    task_id = generate_cache_key("analysis", transcript_text, research_question, analysis_mode, str(framework))
    logger.info(f"Starting analysis task with ID: {task_id}")
    
    if task_id in st.session_state.active_tasks:
        logger.info(f"Task {task_id} already active. Returning existing task ID.")
        return task_id
    
    # Set analysis start time
    st.session_state.analysis_start_time = time.time()
    logger.info(f"Analysis start time set for task {task_id}: {st.session_state.analysis_start_time}")
    
    # Create a queue for this task to send progress updates to the main thread
    progress_queue = Queue()

    # Initialize progress tracking in session state
    st.session_state[f"progress_{task_id}"] = 0.0
    st.session_state[f"status_{task_id}"] = "Starting analysis..."
    st.session_state["analysis_complete"] = False
    st.session_state["analysis_results"] = None
    st.session_state[f"progress_queue_{task_id}"] = progress_queue # Store queue in session state
    logger.info(f"Session state initialized for task {task_id}, progress queue created.")
    
    async def analysis_task():
        final_result = None
        try:
            logger.info(f"Running analysis_task for task {task_id}")
            async for progress, result in process_transcript_async(
                transcript_text, research_question, analysis_mode, framework
            ):
                # Put updates on the queue instead of direct session state modification
                progress_queue.put(("progress", progress, f"Analyzing transcript... {progress:.1%}"))
                logger.debug(f"Task {task_id} progress put to queue: {progress:.1%}")

            final_result = result # The last result yielded when progress is 1.0
            progress_queue.put(("complete_data", final_result))
            logger.info(f"Task {task_id} completed successfully, final result put to queue.")
            
        except Exception as e:
            progress_queue.put(("error", str(e)))
            logger.error(f"Error in analysis task {task_id}: {str(e)}")
            logger.error(traceback.format_exc())
            st.error(f"Error during analysis: {str(e)}") # Display error immediately
            # Do not re-raise here, let the queue signal completion/error

    task_manager.start_task(task_id, analysis_task(), progress_queue) # Pass queue to task manager
    st.session_state.active_tasks.add(task_id)
    logger.info(f"Async task {task_id} handed to task_manager.")
    return task_id

# Add debug information to the UI
if st.sidebar.checkbox("Show Debug Information"):
    st.sidebar.write("### Debug Information")
    st.sidebar.write(f"Cache TTL: {CACHE_TTL} seconds")
    st.sidebar.write(f"Max File Size: {MAX_FILE_SIZE/1024/1024}MB")
    st.sidebar.write(f"Chunk Size: {CHUNK_SIZE} words")
    st.sidebar.write(f"Chunk Overlap: {OVERLAP} words")
    
    if 'analysis_start_time' in st.session_state and st.session_state.analysis_start_time is not None:
        duration = time.time() - st.session_state.analysis_start_time
        st.sidebar.write(f"Last Analysis Duration: {duration:.2f} seconds")

if __name__ == "__main__":
    main()