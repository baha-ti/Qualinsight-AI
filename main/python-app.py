import sys
import streamlit as st
import pandas as pd
import random
import io
# from openai import OpenAI # Not directly used for OpenRouter
import json
from io import BytesIO
from typing import List, Optional, Dict, Any
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

# Constants
VALID_TRANSCRIPT_TYPES = [".txt", ".docx", ".pdf"]
VALID_FRAMEWORK_TYPES = ["json", "txt"]
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat:free"
MAX_TOKENS = 4000
TEMPERATURE = 0.7
CHUNK_SIZE = 4000
OVERLAP = 200

# Get API key from secrets
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Define the headers for the API request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/baha-ti/Qualinsight-AI",
    "X-Title": "Qualinsight AI"
}

def get_ai_response(messages):
    """Get response from DeepSeek API"""
    data = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

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

# --- Title ---\nst.title("QualInsight AI - Qualitative Research Assistant")

# --- Step 1: Input Section ---\nst.header("Step 1: Input")
st.subheader("Upload Transcript")
input_method = st.radio("Choose input method:", ["Upload File", "Paste Text"], key="input_method_radio")

transcript_text = None

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload transcript (.txt, .docx, or .pdf)", type=VALID_TRANSCRIPT_TYPES, key="transcript_uploader")
    if uploaded_file is not None:
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

# Knowledge Base Management (Theories/Frameworks)
if st.session_state.analysis_mode == "Deductive":
    st.subheader("Knowledge Base Management")

    kb_action = st.radio(
        "Choose action:",
        ("Add New", "Use Existing", "Edit Existing", "Delete"),
        key="kb_action_radio"
    )

    # Initialize knowledge base in session state if it doesn't exist
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = load_knowledge_base()

    if kb_action == "Add New":
        with st.form("new_framework_form"):
            kb_name = st.text_input("Enter name for this framework/theory:", key="new_kb_name")
            st.subheader("Framework Structure")
            overview = st.text_area("Overview/Description:", height=100, key="new_overview")
            key_concepts = st.text_area("Key Concepts (one per line):", height=100, key="new_key_concepts")
            methodology = st.text_area("Methodology/Approach:", height=100, key="new_methodology")
            applications = st.text_area("Applications/Use Cases:", height=100, key="new_applications")
            references = st.text_area("References:", height=100, key="new_references")
            
            if st.form_submit_button("Save to Knowledge Base"):
                if kb_name and overview:
                    framework_data = {
                        "overview": overview,
                        "key_concepts": key_concepts,
                        "methodology": methodology,
                        "applications": applications,
                        "references": references,
                        "last_modified": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.session_state.knowledge_base[kb_name] = framework_data
                    save_knowledge_base(st.session_state.knowledge_base)
                    st.success(f"Saved '{kb_name}' to knowledge base!")

    elif kb_action == "Edit Existing":
        if st.session_state.knowledge_base:
            selected_kb = st.selectbox("Select Framework/Theory to Edit:", list(st.session_state.knowledge_base.keys()), key="edit_kb_select")
            framework_data = st.session_state.knowledge_base[selected_kb]
            
            with st.form("edit_framework_form"):
                new_name = st.text_input("Framework/Theory Name:", value=selected_kb, key="edit_kb_name")
                overview = st.text_area("Overview/Description:", value=framework_data.get("overview", ""), height=100, key="edit_overview")
                key_concepts = st.text_area("Key Concepts:", value=framework_data.get("key_concepts", ""), height=100, key="edit_key_concepts")
                methodology = st.text_area("Methodology/Approach:", value=framework_data.get("methodology", ""), height=100, key="edit_methodology")
                applications = st.text_area("Applications/Use Cases:", value=framework_data.get("applications", ""), height=100, key="edit_applications")
                references = st.text_area("References:", value=framework_data.get("references", ""), height=100, key="edit_references")
                
                if st.form_submit_button("Update Framework"):
                    if new_name:
                        # Remove old entry if name changed
                        if new_name != selected_kb:
                            del st.session_state.knowledge_base[selected_kb]
                        
                        framework_data = {
                            "overview": overview,
                            "key_concepts": key_concepts,
                            "methodology": methodology,
                            "applications": applications,
                            "references": references,
                            "last_modified": time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.knowledge_base[new_name] = framework_data
                        save_knowledge_base(st.session_state.knowledge_base)
                        st.success(f"Updated '{new_name}' in knowledge base!")

    elif kb_action == "Delete":
        if st.session_state.knowledge_base:
            selected_kb = st.selectbox("Select Framework/Theory to Delete:", list(st.session_state.knowledge_base.keys()), key="delete_kb_select")
            if st.button("Delete Framework", key="delete_kb_button"):
                del st.session_state.knowledge_base[selected_kb]
                save_knowledge_base(st.session_state.knowledge_base)
                st.success(f"Deleted '{selected_kb}' from knowledge base!")

# Only show framework/theory selection for deductive analysis
if analysis_mode == "Deductive":
    if st.session_state.knowledge_base:
        selected_kb = st.selectbox("Select Framework/Theory:", list(st.session_state.knowledge_base.keys()), key="use_kb_select")
        framework_data = st.session_state.knowledge_base[selected_kb]
        framework_text = f"""
        Framework: {selected_kb}
        Overview: {framework_data.get('overview', '')}
        Key Concepts: {framework_data.get('key_concepts', '')}
        Methodology: {framework_data.get('methodology', '')}
        Applications: {framework_data.get('applications', '')}
        """
    else:
        st.warning("No frameworks/theories in knowledge base. Please add one first.")
        framework_text = ""
else:
    framework_text = ""

# --- Trigger Analysis Button ---\
if st.button("Start Analysis", key="start_analysis_button"):
    if not transcript_text:
        st.error("Please upload or paste a transcript to start analysis.")
    elif not research_questions:
        st.error("Please enter your research questions to start analysis.")
    else:
        # --- Step 2: Processing ---\
        st.header("Step 2: Analysis Results")

        # Stage 1: Initial Coding
        st.subheader("Stage 1: Initial Coding")

        # Process the chunks
        all_results = []
        for i, chunk in enumerate(transcript_text):
            try:
                with st.spinner(f"Processing chunk {i+1} of {len(transcript_text)}..."):
                    # First stage prompt - focus on identifying codes and organizing by respondent
                    system_prompt = """You are an AI assistant analyzing interview transcripts. 
                    For this first stage, focus on:
                    1. Identifying the respondent/speaker (if not already marked)
                    2. Breaking down the text into meaningful segments
                    3. Assigning initial codes to each segment

                    IMPORTANT: Your response MUST be a plain text string, with each coded entry separated by `###CODING_ENTRY###`. Each entry must contain the following fields, delimited by `|||`:
                    - RESPONDENT: <name or identifier of the speaker>
                    - TEXT: <exact quote from transcript>
                    - CODE: <initial code for this segment>
                    - NOTES: <any additional observations>

                    Example response format:
                    RESPONDENT: Participant 1 ||| TEXT: I found the exercise challenging, especially when I had to \"think outside the box\". ||| CODE: Difficulty Level ||| NOTES: Mentioned challenge with specific task and metaphorical thinking.\nAlso noted time constraints.###CODING_ENTRY###
                    RESPONDENT: Participant 2 ||| TEXT: The instructions were clear. ||| CODE: Clarity ||| NOTES: Positive feedback on instructions###CODING_ENTRY###
                    """
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Research Questions: {research_questions}\n\nAnalyze this transcript chunk:\n\n{chunk}"}
                    ]
                    
                    response = get_ai_response(messages)
                    if response:
                        try:
                            all_segment_results = []
                            # Clean the response to ensure it's parsable
                            response = response.replace('```json', '').replace('```', '')
                            response = response.strip()
                            
                            # Split by the entry delimiter
                            entries = response.split('###CODING_ENTRY###')
                            
                            for entry in entries:
                                if not entry.strip():
                                    continue
                                
                                # Parse each field within the entry
                                parts = entry.split(' ||| ')
                                
                                segment_data = {}
                                for part in parts:
                                    if ': ' in part:
                                        key, value = part.split(': ', 1)
                                        segment_data[key.strip().lower()] = value.strip()
                                
                                # Validate and add to results
                                required_fields = ['respondent', 'text', 'code', 'notes']
                                if all(field in segment_data for field in required_fields):
                                    all_segment_results.append(segment_data)
                                else:
                                    st.warning(f"Skipping malformed entry in chunk {i+1}: {entry}")
                            
                            all_results.extend(all_segment_results)
                            
                        except Exception as e:
                            st.error(f"Failed to parse AI response for chunk {i+1}. Error: {str(e)}")
                            st.text("Raw response:")
                            st.text(response)
                            continue
            except Exception as e:
                st.error(f"Error processing chunk {i+1}: {str(e)}")
        
        if all_results:
            # Display Stage 1 Results
            st.subheader("Initial Coding Results")
            
            # Create DataFrame for initial coding
            df_initial = pd.DataFrame(all_results)
            
            # Group by respondent
            if 'respondent' in df_initial.columns:
                st.write("### Coding by Respondent")
                for respondent in df_initial['respondent'].unique():
                    st.write(f"**Respondent: {respondent}**")
                    respondent_df = df_initial[df_initial['respondent'] == respondent]
                    st.dataframe(respondent_df[['text', 'code', 'notes']])
                    st.write("---")
            
            # Stage 2: Theme Development
            st.subheader("Stage 2: Theme Development")
            
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
                        THEME: Learning Experience ||| SUBTHEME: Instructional Clarity ||| CODES: Clear Instructions, Understanding ||| EVIDENCE: The instructions were very clear, I understood what to do, it was \"spot on\" ||| EXPLANATION: Participants found the instructions clear and easy to follow###THEME_ENTRY###
                        """
                        
                        # Convert initial coding to string for analysis
                        coding_summary_str = df_initial.to_string(index=False)
                        
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Research Questions: {research_questions}\n\nInitial Coding:\n{coding_summary_str}"}
                        ]
                        
                        response = get_ai_response(messages)
                        if response:
                            try:
                                all_theme_results = []
                                # Clean the response to ensure it's parsable
                                response = response.replace('```json', '').replace('```', '')
                                response = response.strip()
                                
                                # Split by the entry delimiter
                                entries = response.split('###THEME_ENTRY###')
                                
                                for entry in entries:
                                    if not entry.strip():
                                        continue
                                    
                                    # Parse each field within the entry
                                    parts = entry.split(' ||| ')
                                    
                                    theme_data = {}
                                    for part in parts:
                                        if ': ' in part:
                                            key, value = part.split(': ', 1)
                                            theme_data[key.strip().lower()] = value.strip()
                                    
                                    # Convert comma-separated strings to lists for 'codes' and 'evidence'
                                    if 'codes' in theme_data: theme_data['codes'] = [c.strip() for c in theme_data['codes'].split(',') if c.strip()]
                                    if 'evidence' in theme_data: theme_data['evidence'] = [e.strip() for e in theme_data['evidence'].split(',') if e.strip()]
                                    
                                    # Validate and add to results
                                    required_fields = ['theme', 'subtheme', 'codes', 'evidence', 'explanation']
                                    if all(field in theme_data for field in required_fields):
                                        all_theme_results.append(theme_data)
                                    else:
                                        st.warning(f"Skipping malformed entry in theme development: {entry}")
                                
                                # Create DataFrame for themes
                                df_themes = pd.DataFrame(all_theme_results)
                                
                                # Display Theme Analysis
                                st.write("### Theme Analysis")
                                st.dataframe(df_themes)
                                
                                # Create highlighted transcript
                                st.subheader("Highlighted Transcript with Themes")
                                highlighted_text = transcript_text[0]  # Assuming single transcript
                                
                                # Define colors for themes
                                theme_colors = {
                                    theme: f"🔹[{theme}]" for theme in df_themes['theme'].unique()
                                }
                                
                                # Apply highlights
                                for _, row in df_themes.iterrows():
                                    for quote in row['evidence']:
                                        # Remove potential surrounding quotes from evidence string
                                        clean_quote = quote.strip('\'"')
                                        if clean_quote in highlighted_text:
                                            theme = row['theme']
                                            highlight = theme_colors[theme]
                                            highlighted_text = highlighted_text.replace(clean_quote, f"{highlight} {clean_quote}", 1)
                                
                                st.text_area("Highlighted Transcript", highlighted_text, height=400)
                                
                                # Export Options
                                st.subheader("Export Options")
                                
                                # Export initial coding
                                csv_initial = df_initial.to_csv(index=False)
                                st.download_button(
                                    "Download Initial Coding (CSV)",
                                    csv_initial,
                                    "initial_coding.csv",
                                    "text/csv",
                                    key="download_initial_csv"
                                )
                                
                                # Export theme analysis
                                csv_themes = df_themes.to_csv(index=False)
                                st.download_button(
                                    "Download Theme Analysis (CSV)",
                                    csv_themes,
                                    "theme_analysis.csv",
                                    "text/csv",
                                    key="download_themes_csv"
                                )
                                
                                # Export full report
                                doc = Document()
                                doc.add_heading('Qualitative Analysis Report', 0)
                                doc.add_paragraph('Research Questions: ' + research_questions)
                                
                                # Add initial coding
                                doc.add_heading('Initial Coding', level=1)
                                for respondent in df_initial['respondent'].unique():
                                    doc.add_heading(f'Respondent: {respondent}', level=2)
                                    respondent_df = df_initial[df_initial['respondent'] == respondent]
                                    table = doc.add_table(rows=1, cols=3)
                                    hdr = table.rows[0].cells
                                    hdr[0].text, hdr[1].text, hdr[2].text = 'Text', 'Code', 'Notes'
                                    for _, row in respondent_df.iterrows():
                                        row_cells = table.add_row().cells
                                        row_cells[0].text = str(row['text'])
                                        row_cells[1].text = str(row['code'])
                                        row_cells[2].text = str(row['notes'])
                                
                                # Add theme analysis
                                doc.add_heading('Theme Analysis', level=1)
                                table = doc.add_table(rows=1, cols=5)
                                hdr = table.rows[0].cells
                                hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text, hdr[4].text = 'Theme', 'Subtheme', 'Codes', 'Evidence', 'Explanation'
                                for _, row in df_themes.iterrows():
                                    row_cells = table.add_row().cells
                                    row_cells[0].text = str(row['theme'])
                                    row_cells[1].text = str(row['subtheme'])
                                    row_cells[2].text = str(row['codes'])
                                    row_cells[3].text = str(row['evidence'])
                                    row_cells[4].text = str(row['explanation'])
                                
                                # Add highlighted transcript
                                doc.add_heading('Highlighted Transcript', level=1)
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
                                st.error(f"Failed to parse theme development response. Error: {str(e)}")
                                st.text("Raw response:")
                                st.text(response)
                except Exception as e:
                    st.error(f"Error in theme development: {str(e)}")


def process_transcript(uploaded_file) -> Optional[List[str]]:
    """Process uploaded transcript files (TXT, DOCX, PDF)"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    text_content = None

    if file_type == "txt":
        text_content = extract_text_from_txt(uploaded_file)
    elif file_type == "docx":
        text_content = extract_text_from_docx(uploaded_file)
    elif file_type == "pdf":
        text_content = extract_text_from_pdf(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return None

    if text_content:
        # Simple chunking for now, can be improved
        chunks = chunk_text(text_content)
        return chunks
    return None

def extract_text_from_txt(file) -> str:
    """Extracts text from a .txt file."""
    return file.read().decode('utf-8')

def extract_text_from_docx(file) -> str:
    """Extracts text from a .docx file."""
    doc = Document(file)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def extract_text_from_pdf(file) -> str:
    """Extracts text from a .pdf file."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name
    try:
        text = pdf_extract_text(tmp_file_path)
    finally:
        os.remove(tmp_file_path)
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Chunks text into smaller pieces for API processing."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        
        # Try to find a natural break point (e.g., end of sentence)
        break_point = text.rfind('.', start, end)
        if break_point == -1 or break_point < start + chunk_size * 0.8: # Ensure chunk is not too small
            break_point = end - 1 # Fallback to hard cut
        
        chunks.append(text[start : break_point + 1])
        start = break_point + 1 - OVERLAP # Overlap chunks
        if start < 0: start = 0 # Ensure start doesn't go below 0
    return chunks

def ai_generate_codes(text: str, mode: str, rq: str, framework: Optional[str] = None) -> List[dict]:
    """Generates initial codes or themes using the AI."""
    # This function is not used directly anymore, but is kept for reference or future use.
    # The prompt building and API call logic is now directly in the main app flow for more control.
    pass