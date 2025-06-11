import sys
import streamlit as st
import pandas as pd
import re
import random
import io
from openai import OpenAI
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
import openai
import requests
import os

# Constants
VALID_TRANSCRIPT_TYPES = [".txt", ".docx", ".pdf"]
VALID_FRAMEWORK_TYPES = ["json", "txt"]
API_URL = 'https://openrouter.ai/api/v1/chat/completions'
MODEL = "deepseek/deepseek-chat:free"
MAX_TOKENS = 4000
TEMPERATURE = 0.7
CHUNK_SIZE = 4000
OVERLAP = 200

# Get API key from secrets
API_KEY = st.secrets["OPENROUTER_API_KEY"]

# Define the headers for the API request
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json',
    'HTTP-Referer': 'https://github.com/baha-ti/Qualinsight-AI',
    'X-Title': 'Qualinsight AI'
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

# --- Title ---
st.title("QualInsight AI - Qualitative Research Assistant")

# --- Step 1: Input Section ---
st.header("Step 1: Input")
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
st.subheader("Research Questions")
research_questions = st.text_area("Enter your research question(s):", height=100, key="research_questions")

# Analysis Settings
st.subheader("Analysis Settings")
analysis_mode = st.radio("Choose analysis approach:", ["Inductive", "Deductive"], key="analysis_mode_radio")

# Knowledge Base Management
st.subheader("Knowledge Base Management")
kb_action = st.radio("Knowledge Base Action:", ["Use Existing", "Add New", "Edit Existing", "Delete"], key="kb_action_radio")

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

# --- Step 2: Processing ---
if transcript_text and research_questions:
    if st.button("Start Analysis", key="start_analysis"):
        st.header("Step 2: Analysis Results")
        
        # Process the chunks
        all_results = []
        for i, chunk in enumerate(transcript_text):
            try:
                with st.spinner(f"Processing chunk {i+1} of {len(transcript_text)}..."):
                    system_prompt = """You are an AI assistant analyzing interview transcripts. 
                    For each segment of text, identify:
                    1. Code (specific category)
                    2. Subtheme (subcategory)
                    3. Theme (main category)
                    4. The exact text segment that supports this coding
                    
                    Format your response as a JSON array of objects with these fields:
                    {
                        "text": "exact quote from transcript",
                        "code": "specific category",
                        "subtheme": "subcategory",
                        "theme": "main category"
                    }
                    """
                    
                    if analysis_mode == "Deductive":
                        system_prompt += f"\nUse this framework to guide your analysis:\n{framework_text}"
                    else:
                        system_prompt += "\nProvide an inductive analysis, identifying emerging themes and patterns."
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Research Questions: {research_questions}\n\nAnalyze this transcript chunk:\n\n{chunk}"}
                    ]
                    
                    response = get_ai_response(messages)
                    if response:
                        try:
                            # Parse the JSON response
                            codes = json.loads(response)
                            all_results.extend(codes)
                        except json.JSONDecodeError:
                            st.error(f"Failed to parse AI response for chunk {i+1}")
                            continue
            except Exception as e:
                st.error(f"Error processing chunk {i+1}: {str(e)}")
        
        if all_results:
            # --- Step 3: Output Display ---
            st.header("Step 3: Analysis Output")
            
            # Create DataFrame
            df = pd.DataFrame(all_results)
            
            # Display the coding table
            st.subheader("Coding Table")
            st.dataframe(df)
            
            # Create highlighted transcript
            st.subheader("Highlighted Transcript")
            highlighted_text = transcript_text[0]  # Assuming single transcript
            for _, row in df.iterrows():
                text = row['text']
                code = row['code']
                theme = row['theme']
                highlight = f"🔹[{code} → {theme}]"
                highlighted_text = highlighted_text.replace(text, f"{highlight} {text}", 1)
            
            st.text_area("Highlighted Transcript", highlighted_text, height=400)
            
            # --- Step 4: Export Options ---
            st.header("Step 4: Export Options")
            
            # Export as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "Download Coding Table (CSV)",
                csv,
                "coding_table.csv",
                "text/csv",
                key="download_csv"
            )
            
            # Export as Word
            doc = Document()
            doc.add_heading('Qualitative Analysis Report', 0)
            doc.add_paragraph('Research Questions: ' + research_questions)
            doc.add_heading('Coding Table', level=1)
            table = doc.add_table(rows=1, cols=4)
            hdr = table.rows[0].cells
            hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text = 'Text', 'Code', 'Subtheme', 'Theme'
            for _, row in df.iterrows():
                row_cells = table.add_row().cells
                row_cells[0].text = str(row['text'])
                row_cells[1].text = str(row['code'])
                row_cells[2].text = str(row['subtheme'])
                row_cells[3].text = str(row['theme'])
            
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
            
            # Export as PDF
            pdf_stream = BytesIO()
            pdf = SimpleDocTemplate(pdf_stream, pagesize=letter)
            styles = getSampleStyleSheet()
            elems = [
                Paragraph('Qualitative Analysis Report', styles['Title']),
                Paragraph('Research Questions: ' + research_questions, styles['Normal']),
                Spacer(1, 12),
                Paragraph('Coding Table', styles['Heading1']),
                Table([['Text', 'Code', 'Subtheme', 'Theme']] + df.values.tolist()),
                Spacer(1, 12),
                Paragraph('Highlighted Transcript', styles['Heading1']),
                Paragraph(highlighted_text, styles['Normal'])
            ]
            pdf.build(elems)
            pdf_stream.seek(0)
            st.download_button(
                "Download Full Report (PDF)",
                pdf_stream,
                "analysis_report.pdf",
                "application/pdf",
                key="download_pdf"
            )

# --- Sidebar for API Key ---
# Use the API key from Streamlit secrets
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.base_url = "https://openrouter.ai/api/v1"

# Initialize OpenAI client with OpenRouter configuration
client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/baha-ti/Qualinsight-AI",  # Your app's URL
        "X-Title": "Qualinsight AI"  # Your app's name
    }
)

st.sidebar.info(f"Python executable: {sys.executable}")

# --- Upload section ---
uploaded_file = st.file_uploader("Upload transcript (.txt, .docx, or .pdf)", type=VALID_TRANSCRIPT_TYPES)
framework_file = st.file_uploader("Upload framework for deductive coding (.json or .txt)", type=VALID_FRAMEWORK_TYPES)

# --- Extract text functions ---
def extract_text_from_txt(file) -> str:
    try:
        return file.read().decode("utf-8")
    except UnicodeDecodeError:
        file.seek(0)
        return file.read().decode("latin-1")


def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])


def extract_text_from_pdf(file) -> str:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file.seek(0)
        return pdf_extract_text(temp_file.name)


def extract_text(file) -> str:
    if not file:
        return ""
    try:
        file.seek(0)
        ext = file.name.split('.')[-1].lower()
        if ext == 'txt':
            return extract_text_from_txt(file)
        elif ext == 'docx':
            return extract_text_from_docx(file)
        elif ext == 'pdf':
            return extract_text_from_pdf(file)
        else:
            st.error(f"Unsupported file type: {ext}")
            return ""
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        return ""

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into smaller chunks."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        if current_size + word_size > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# --- AI-based code generator ---
def ai_generate_codes(text: str, mode: str, rq: str, framework: Optional[str] = None) -> List[dict]:
    if not client:
        st.error("Please configure a valid OpenAI API Key in the sidebar.")
        return []

    system_msg = (
        f"You are a qualitative research assistant trained in thematic analysis. Apply mode: {mode}."
    )
    if framework:
        system_msg += f" Use this framework for deductive coding: {framework}"

    # Split text into chunks
    chunks = chunk_text(text)
    all_results = []
    
    for i, chunk in enumerate(chunks):
        try:
            with st.spinner(f"Processing chunk {i+1} of {len(chunks)}..."):
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": (
                        f"Research Questions:\n{rq}\n\n"
                        f"Framework:\n{framework or 'None'}\n\n"
                        f"Transcript Chunk {i+1}/{len(chunks)}:\n{chunk}"
                    )}
                ]
                
                response = get_ai_response(messages)
                if response:
                    st.write(f"Analysis for chunk {i+1}:")
                    st.write(response)
                    st.write("---")
                    all_results.append(response)
        except Exception as e:
            st.error(f"Error processing chunk {i+1}: {str(e)}")
    
    return all_results

# --- Process file and framework ---
if uploaded_file:
    transcript_text = extract_text(uploaded_file)
    if not transcript_text:
        st.error("Transcript file is empty or could not be read.")
    else:
        st.subheader("Transcript Preview")
        st.text_area("Transcript", transcript_text, height=200)

        # Load framework if provided
        framework_text = None
        if framework_file:
            raw = extract_text(framework_file)
            if framework_file.name.endswith(".json"):
                try:
                    framework_json = json.loads(raw)
                    framework_text = json.dumps(framework_json)
                except json.JSONDecodeError:
                    st.warning("Framework file is not valid JSON. Using raw text.")
                    framework_text = raw
            else:
                framework_text = raw

        if st.button("Generate AI Codes and Highlights"):
            with st.spinner("Analyzing with AI..."):
                results = ai_generate_codes(transcript_text, analysis_mode, research_questions, framework_text)
            if results:
                st.success("Analysis complete!")
                st.write("### Complete Analysis")
                for i, result in enumerate(results):
                    st.write(f"**Chunk {i+1} Analysis:**")
                    st.write(result)
                    st.write("---")
            else:
                st.warning("No results returned by the AI. Please check your input and try again.")
        else:
            st.warning("No results returned by the AI. Please check your input and try again.")