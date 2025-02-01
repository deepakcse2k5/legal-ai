import gradio as gr
import os
import shutil
import pdfplumber  # Ensure this is installed
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

# Import utilities from modules (Make sure files exist in the same directory)
from src.utils import load_pdf, create_retriever, get_qa_chain, summarize_document
from src.generate_report import extract_text_from_pdf, generate_complaint_report, save_report_as_docx
import logging
logging.basicConfig(level=logging.DEBUG)


# Thread pool for handling CPU-heavy tasks
executor = ThreadPoolExecutor()

# Store the latest document processing state
latest_state = {
    "docs": None,
    "retriever": None,
    "qa_chain": None,
    "summary": None
}


# ✅ Function to Upload and Process PDF
def upload_file(file):
    try:
        temp_filename = "latest_uploaded.pdf"
        with open(temp_filename, "wb") as buffer:
            buffer.write(file.read())

        # Process PDF
        docs = load_pdf(temp_filename)
        retriever = create_retriever(docs)
        qa_chain = get_qa_chain(retriever)

        latest_state["docs"] = docs
        latest_state["retriever"] = retriever
        latest_state["qa_chain"] = qa_chain

        os.remove(temp_filename)  # Cleanup
        return "File uploaded and processed successfully"
    except Exception as e:
        return f"Upload failed: {str(e)}"


# ✅ Function to Get Summary
def get_summary():
    if latest_state["docs"] is None:
        return "No document uploaded."
    if latest_state["summary"] is None:
        latest_state["summary"] = summarize_document(latest_state["docs"])
    return latest_state["summary"]


# ✅ Function to Ask AI Legal Assistant
def ask_question(question):
    if latest_state["qa_chain"] is None:
        return "No document uploaded."
    response = latest_state["qa_chain"](question)["result"]
    return response


# ✅ Function to Upload Reference PDF
def upload_pdf(file):
    try:
        filepath = f"./{file.name}"
        with open(filepath, "wb") as buffer:
            buffer.write(file.read())

        reference_text = extract_text_from_pdf(filepath)
        return "File uploaded successfully", reference_text
    except Exception as e:
        return f"Reference file upload failed: {str(e)}", ""


# ✅ Function to Generate Legal Report
def generate_report(case_title, court, petitioner, respondent, filing_date, incident_summary, legal_claims,
                    reliefs_sought, evidence, reference_text):
    try:
        if not reference_text:
            return "Reference text is required."

        case_details = {
            "case_title": case_title,
            "court": court,
            "petitioner": petitioner,
            "respondent": respondent,
            "filing_date": filing_date,
            "incident_summary": incident_summary,
            "legal_claims": legal_claims,
            "reliefs_sought": reliefs_sought,
            "evidence": evidence,
            "reference_text": reference_text
        }

        report_text = generate_complaint_report(reference_text, case_details)
        filename = f"Complaint_Report_{petitioner.replace(' ', '_')}.docx"
        filepath = save_report_as_docx(report_text, filename)

        return f"Report generated successfully. Download: {filepath}"
    except Exception as e:
        return f"Report generation failed: {str(e)}"


# ✅ Creating the Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 📑 AI-Powered Legal Document Processing System")

    with gr.Tab("📂 Upload Document"):
        upload_button = gr.File(label="Upload PDF")
        upload_output = gr.Textbox()
        upload_button.upload(upload_file, upload_button, upload_output)

    with gr.Tab("📖 Get Summary"):
        summary_button = gr.Button("Get Summary")
        summary_output = gr.Textbox()
        summary_button.click(get_summary, inputs=[], outputs=summary_output)

    with gr.Tab("❓ Ask AI Legal Assistant"):
        question_input = gr.Textbox(label="Enter your question")
        question_button = gr.Button("Ask")
        question_output = gr.Textbox()
        question_button.click(ask_question, inputs=[question_input], outputs=question_output)

    with gr.Tab("📤 Upload Reference PDF"):
        ref_upload_button = gr.File(label="Upload Reference PDF")
        ref_upload_output = gr.Textbox()
        ref_upload_text = gr.Textbox()
        ref_upload_button.upload(upload_pdf, ref_upload_button, [ref_upload_output, ref_upload_text])

    with gr.Tab("📜 Generate Report"):
        case_title = gr.Textbox(label="Case Title")
        court = gr.Textbox(label="Court")
        petitioner = gr.Textbox(label="Petitioner")
        respondent = gr.Textbox(label="Respondent")
        filing_date = gr.Textbox(label="Filing Date")
        incident_summary = gr.Textbox(label="Incident Summary")
        legal_claims = gr.Textbox(label="Legal Claims")
        reliefs_sought = gr.Textbox(label="Reliefs Sought")
        evidence = gr.Textbox(label="Evidence")
        reference_text = gr.Textbox(label="Reference Text")
        generate_button = gr.Button("Generate Report")
        report_output = gr.Textbox()

        generate_button.click(
            generate_report,
            inputs=[case_title, court, petitioner, respondent, filing_date, incident_summary, legal_claims,
                    reliefs_sought, evidence, reference_text],
            outputs=report_output
        )

if __name__ == "__main__":
    print("🚀 Starting Gradio App...")
    demo.launch(share=True, server_port=7860)

