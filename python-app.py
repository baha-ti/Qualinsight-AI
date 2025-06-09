import streamlit as st
import pandas as pd
import re
import random
import io
import openai
import json
from io import BytesIO
from typing import List, Optional
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from pdfminer.high_level import extract_text as pdf_extract_text
import tempfile

# Constants
VALID_TRANSCRIPT_TYPES = ["txt", "docx", "pdf"]
VALID_FRAMEWORK_TYPES = ["json", "txt"]
OPENAI_MODEL = "gpt-4"
MAX_TOKENS = 1500
TEMPERATURE = 0.5

# --- Title ---
st.title("QualInsight AI - Qualitative Research Assistant")
st.markdown("Upload your transcript, frameworks, and let the AI generate codes, themes, highlights, and reports.")

# --- Sidebar for API Key ---
st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key format."""
    return api_key.startswith("sk-") if api_key else False

if not openai_api_key:
    st.sidebar.warning("Enter your OpenAI API Key to enable AI integration.")

# --- Upload section ---
uploaded_file = st.file_uploader("Upload transcript (.txt, .docx, or .pdf)", type=VALID_TRANSCRIPT_TYPES)
framework_file = st.file_uploader("Upload framework for deductive coding (.json or .txt)", type=VALID_FRAMEWORK_TYPES)

# --- Research question input ---
research_questions = st.text_area("Enter your research question(s)", height=100)

# --- Analysis mode ---
analysis_mode = st.selectbox("Choose coding mode", ["Inductive", "Deductive", "Hybrid"])

# --- Extract text ---
def extract_text_from_txt(file) -> str:
    """Extract text from a TXT file with fallback encoding."""
    try:
        return file.read().decode("utf-8")
    except UnicodeDecodeError:
        file.seek(0)
        return file.read().decode("latin-1")

def extract_text_from_docx(file) -> str:
    """Extract text from a DOCX file."""
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file using temporary file."""
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file.seek(0)
        return pdf_extract_text(temp_file.name)

def extract_text(file) -> str:
    """Extract text from various file types."""
    if not file:
        return ""
    
    try:
        file.seek(0)
        file_extension = file.name.split(".")[-1].lower()
        
        if file_extension == "txt":
            return extract_text_from_txt(file)
        elif file_extension == "docx":
            return extract_text_from_docx(file)
        elif file_extension == "pdf":
            return extract_text_from_pdf(file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return ""
    except Exception as e:
        st.error(f"Failed to extract text: {str(e)}")
        return ""

# --- AI-based code generator ---
def ai_generate_codes(text: str, mode: str, rq: str, framework: Optional[str] = None, api_key: Optional[str] = None) -> List[dict]:
    """Generate codes using OpenAI API."""
    if not validate_api_key(api_key):
        st.error("Invalid OpenAI API Key. Please enter a valid key.")
        return []

    system_msg = (
        "You are a qualitative research assistant trained in thematic analysis. "
        f"Apply the following mode: {mode}."
    )
    if framework:
        system_msg += f" Use this framework for deductive coding: {framework}"

    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Research Questions:\n{rq}\n\nFramework:\n{framework or 'None'}\n\nTranscript:\n{text}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            api_key=api_key
        )
        
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        
        if not isinstance(data, list):
            st.error("AI output is not a list. Check your prompt and input.")
            return []
            
        return data
        
    except json.JSONDecodeError:
        st.error("Failed to parse AI response. Check API output format.\n\nRaw output:\n" + content)
        return []
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return []

# --- Process file and framework ---
if uploaded_file:
    transcript_text = extract_text(uploaded_file)
    if not transcript_text:
        st.error("Transcript file is empty or could not be read.")
    else:
        st.subheader("Transcript Preview")
        st.text_area("Transcript", transcript_text, height=200)
        framework_text = None
        if framework_file:
            raw = extract_text(framework_file)
            # If JSON, load
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
            if not openai_api_key:
                st.error("Provide OpenAI API Key in sidebar to proceed.")
            else:
                with st.spinner("Analyzing with AI..."):
                    results = ai_generate_codes(transcript_text, analysis_mode, research_questions, framework_text, openai_api_key)
                if results:
                    # Validate for required keys
                    filtered = []
                    for r in results:
                        if all(k in r for k in ("Text", "Code", "Theme")):
                            filtered.append(r)
                    if not filtered:
                        st.warning("AI did not return the expected fields.")
                    df = pd.DataFrame(filtered)
                    st.subheader("Coded Segments")
                    st.dataframe(df)

                    # Highlight transcript (avoid replacing text inside text)
                    st.subheader("Highlighted Transcript")
                    highlights = transcript_text
                    used = set()
                    for row in filtered:
                        code_text = row['Text']
                        tag = f"🔹[{row['Code']}]"
                        # Only highlight first occurrence of each code_text
                        if code_text and code_text not in used:
                            highlights = highlights.replace(code_text, f"{tag} {code_text}", 1)
                            used.add(code_text)
                    st.text_area("Highlights", highlights, height=300)

                    # --- Download code table ---
                    csv = df.to_csv(index=False)
                    st.download_button("Download Code Table as CSV", csv, "coded_segments.csv", "text/csv")

                    # --- Generate Word report ---
                    doc = Document()
                    doc.add_heading('Qualitative Analysis Report', 0)
                    doc.add_paragraph('Research Questions: ' + research_questions)
                    doc.add_heading('Codes and Themes', level=1)
                    table = doc.add_table(rows=1, cols=3)
                    hdr_cells = table.rows[0].cells
                    hdr_cells[0].text = 'Text'
                    hdr_cells[1].text = 'Code'
                    hdr_cells[2].text = 'Theme'
                    for _, r in df.iterrows():
                        row_cells = table.add_row().cells
                        row_cells[0].text = str(r['Text'])
                        row_cells[1].text = str(r['Code'])
                        row_cells[2].text = str(r['Theme'])
                    doc.add_page_break()
                    doc_stream = BytesIO()
                    doc.save(doc_stream)
                    doc_stream.seek(0)
                    st.download_button("Download Word Report", doc_stream, "report.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

                    # --- Generate PDF report ---
                    pdf_stream = BytesIO()
                    pdf = SimpleDocTemplate(pdf_stream, pagesize=letter)
                    styles = getSampleStyleSheet()
                    elems = []
                    elems.append(Paragraph('Qualitative Analysis Report', styles['Title']))
                    elems.append(Paragraph('Research Questions: ' + research_questions, styles['Normal']))
                    elems.append(Spacer(1, 12))
                    # Table data
                    table_data = [['Text', 'Code', 'Theme']] + df.values.tolist()
                    pdf_table = Table(table_data)
                    elems.append(pdf_table)
                    pdf.build(elems)
                    pdf_stream.seek(0)
                    st.download_button("Download PDF Report", pdf_stream, "report.pdf", "application/pdf")
                else:
                    st.warning("No codes/themes generated.")
