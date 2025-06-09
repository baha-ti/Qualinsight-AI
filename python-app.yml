import streamlit as st
import pandas as pd
import docx2txt
import fitz  # PyMuPDF
import re
import random
import io
import openai
import json
from io import BytesIO
from typing import List
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- Title ---
st.title("Qualitative Data Analysis Tool with AI Integration")
st.markdown("Upload your transcript, frameworks, and let the AI generate codes, themes, highlights, and reports.")

# --- Sidebar for API Key ---
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if openai_api_key:
    openai.api_key = openai_api_key
else:
    st.sidebar.warning("Enter your OpenAI API Key to enable AI integration.")

# --- Upload section ---
uploaded_file = st.file_uploader("Upload transcript (.txt, .docx, or .pdf)", type=["txt", "docx", "pdf"])
framework_file = st.file_uploader("Upload framework for deductive coding (.json or .txt)", type=["json", "txt"])

# --- Research question input ---
research_questions = st.text_area("Enter your research question(s)", height=100)

# --- Analysis mode ---
analysis_mode = st.selectbox("Choose coding mode", ["Inductive", "Deductive", "Hybrid"])

# --- Extract text ---
def extract_text(file) -> str:
    if not file:
        return ""
    try:
        file.seek(0)
        if file.name.endswith(".txt"):
            try:
                return file.read().decode("utf-8")
            except UnicodeDecodeError:
                file.seek(0)
                return file.read().decode("latin-1")
        elif file.name.endswith(".docx"):
            # Save to a temp path for docx2txt
            temp_path = f"/tmp/{random.randint(1,1e9)}.docx"
            with open(temp_path, "wb") as f:
                f.write(file.read())
            return docx2txt.process(temp_path)
        elif file.name.endswith(".pdf"):
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
            return "".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Failed to extract text: {str(e)}")
        return ""
    return ""

# --- AI-based code generator ---
def ai_generate_codes(text: str, mode: str, rq: str, framework: str = None) -> List[dict]:
    system_msg = (
        "You are a qualitative research assistant trained in thematic analysis. "
        "Apply the following mode: " + mode + "."
    )
    if framework:
        system_msg += " Use this framework for deductive coding: " + framework
    prompt = (
        system_msg +
        f"\nResearch Questions:\n{rq}\n\nTranscript:\n" + text +
        "\n\nPlease output a JSON array of objects with fields: Text, Code, Theme. "
        "Generate codes and themes."
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Research Questions:\n{rq}\n\nFramework:\n{framework or 'None'}\n\nTranscript:\n{text}"}
            ],
            temperature=0.5,
            max_tokens=1500
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
        return []
    # Try to extract JSON from the response
    try:
        json_start = content.find('[')
        json_end = content.rfind(']')
        if json_start != -1 and json_end != -1:
            json_str = content[json_start:json_end+1]
            data = json.loads(json_str)
        else:
            data = json.loads(content)
    except Exception:
        st.error("Failed to parse AI response. Check API output format.\n\nRaw output:\n" + content)
        return []
    # Validate parsed data
    if not isinstance(data, list):
        st.error("AI output is not a list. Check your prompt and input.")
        return []
    return data

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
                    results = ai_generate_codes(transcript_text, analysis_mode, research_questions, framework_text)
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
