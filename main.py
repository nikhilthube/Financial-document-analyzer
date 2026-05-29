# Financial Document Analyzer
#from langchain.llms import OpenAI

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

import streamlit as st
import os
import pathlib
import textwrap
from PIL import Image
import fitz  # PyMuPDF for PDF processing

from google import genai
from google.genai import errors, types

# Check if API key is available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY or GEMINI_API_KEY not found. Please set your API key in environment variables.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL_CANDIDATES = [
    os.getenv("GEMINI_MODEL"),
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

## Function to load Gemini model and get responses

def get_gemini_response(input,image_parts,prompt):
    # The API expects a list where the prompt and images are sequential items
    request_content = [prompt] + image_parts + [input]

    last_error = None
    for model_name in MODEL_CANDIDATES:
        if not model_name:
            continue
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=request_content,
            )
            return response.text
        except errors.APIError as exc:
            last_error = exc
            if getattr(exc, "code", None) == 404:
                continue
            raise
        except Exception as exc:
            last_error = exc

    raise last_error
    

def input_image_setup(uploaded_file):
    """
    Processes the uploaded file (image or PDF) and prepares it for the model.
    """
    if uploaded_file is None:
        raise FileNotFoundError("No file was uploaded. Please upload an image or PDF.")

    # Get file extension and bytes
    file_bytes = uploaded_file.getvalue()
    mime_type = uploaded_file.type
    
    image_parts = []

    if mime_type.startswith("image/"):
        # Handle standard image formats
        image_parts.append(types.Part.from_bytes(data=file_bytes, mime_type=mime_type))
        
    elif mime_type == "application/pdf":
        # Handle PDF files by converting each page to an image
        pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            # Render page to a PNG image in memory
            pix = page.get_pixmap()
            img_bytes = pix.tobytes("png")
            image_parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
            
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")
        
    return image_parts


##initialize our streamlit app

st.set_page_config(page_title="Financial Document Analyzer", layout="wide")

st.header("🔍 Financial Document Analyzer powered by Gemini")
st.markdown("---")

# Sidebar for document type selection
with st.sidebar:
    st.header("📋 Document Type")
    doc_type = st.selectbox(
        "Select the type of document you're analyzing:",
        ["Auto-detect", "Invoice/Receipt", "Financial Statement", "Annual Report", "Other Financial Document"]
    )
    
    st.markdown("---")
    st.markdown("**💡 Tips for better analysis:**")
    st.markdown("- Ensure the document is clearly visible")
    st.markdown("- For financial statements, include key sections")
    st.markdown("- Ask specific questions for calculations")
    st.markdown("- PDFs will analyze all pages automatically")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a financial document...", 
        type=["jpg", "jpeg", "png", "pdf"],
        help="Supported formats: JPG, JPEG, PNG, PDF"
    )

with col2:
    st.subheader("❓ Ask Questions")
    input = st.text_input(
        "Ask a question about the document:", 
        key="input",
        placeholder="e.g., What is the P/E ratio? Calculate total revenue for 2024..."
    )
    
    # Example questions based on document type
    if doc_type == "Financial Statement":
        st.markdown("**Example questions:**")
        st.markdown("- What is the P/E ratio for 2024?")
        st.markdown("- Calculate the debt-to-equity ratio")
        st.markdown("- What was the net profit margin?")
        st.markdown("- Show me the revenue growth from 2023 to 2024")
    elif doc_type == "Invoice/Receipt":
        st.markdown("**Example questions:**")
        st.markdown("- What is the total amount due?")
        st.markdown("- Calculate the tax amount")
        st.markdown("- What are the payment terms?")
        st.markdown("- When is this invoice due?")

# Display uploaded file
image = ""   
if uploaded_file is not None:
    try:
        if uploaded_file.type == "application/pdf":
            st.markdown("---")
            st.subheader("📄 Uploaded PDF Document")
            st.info("📋 PDF uploaded successfully! The model will analyze all pages of your document.")
            
            # Show PDF info
            file_bytes = uploaded_file.getvalue()
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            st.success(f"📊 PDF contains {len(pdf_document)} page(s)")
            
        else:
            image = Image.open(uploaded_file)
            st.markdown("---")
            st.subheader("📄 Uploaded Document")
            st.image(image, caption="Uploaded Document", use_container_width=True)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Submit button
submit = st.button("🚀 Analyze Document", type="primary", use_container_width=True)

# Enhanced prompt for financial analysis
input_prompt = """
You are an expert AI assistant specializing in financial document analysis. Your primary role is to accurately answer any question about the provided financial document, whether the question requires simple data extraction or complex financial calculations.

---
### **Rule 0: Be Direct and Concise**
- Your primary goal is to provide a direct answer to the user's specific question. 
- Provide details if the user asks for a summary or details.
- If the user asks for the date or total, provide ONLY the date or total amount.


---
### **Rule 1: Document Validation**
First, verify the image is a financial document. If the uploaded image is NOT a financial document (e.g., memes, general images, social media posts), you MUST respond with EXACTLY this text and nothing else:

"I'm sorry, but this appears to be a non-financial document. I can only analyze financial documents such as:
- Invoices and Receipts
- Financial Statements (Income Statements, Balance Sheets, Cash Flow Statements)
- Annual Reports and Quarterly Reports
- Budget Documents and Forecasts
- Tax Documents and Bank Statements

Please upload a financial document for analysis."

---
### **Rule 2: Answering User Questions**
If the document is financial, your task is to answer the user's question based on its content. Your capabilities include:

**A. Simple Data Extraction (Primary Task):**
- This is your most common task. You will find and report specific pieces of information from the document.
- **Examples:** "What is the invoice date?", "What is the total amount due?", "List the item descriptions.", "Who is the customer?"

**B. Financial Calculations (Advanced Task):**
- If the user's question requires a calculation, perform it accurately using data from the document.
- **Examples:**
  - Gross Margin = (Revenue - Cost of Goods Sold) / Revenue
  - Net Profit Margin = Net Income / Revenue
  - Current Ratio = Current Assets / Current Liabilities
  - And any other requested financial metric.

---
### **Rule 3: Response Guidelines**
- **Synthesize Across Pages:** If multiple images (like PDF pages) are provided, treat them as one continuous document.
- **Show Your Work:** For calculations, briefly explain the formula and the values you used (e.g., "Calculated Gross Margin using Revenue of $1,000 and COGS of $400").
- **Be Factual:** Base your entire response ONLY on the information visible in the provided image(s). Do not use external knowledge.
- **Handle Missing Information:** If the document doesn't contain the data needed to answer, clearly state what information is missing.
- **Format Numbers Clearly:** Use appropriate formatting like commas, currency symbols, and percentages.
"""

## If analyze button is clicked

if submit:
    if uploaded_file is not None:
        try:
            with st.spinner("🔍 Analyzing document..."):
                image_data = input_image_setup(uploaded_file)
                
                # Show processing info for PDFs
                if uploaded_file.type == "application/pdf":
                    st.info(f"📖 Processing {len(image_data)} page(s) from your PDF...")
                
                response = get_gemini_response(input_prompt, image_data, input)
                
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Display response in a nice format
                st.markdown(response)
                
                # Add a download button for the response
                st.download_button(
                    label="💾 Download Analysis",
                    data=response,
                    file_name=f"financial_analysis_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("Please ensure the document is clear and try again.")
    else:
        st.error("⚠️ Please upload a document to analyze.")
        st.info("Supported formats: JPG, JPEG, PNG, PDF")

# Footer
st.markdown("---")
st.markdown("*Powered by Google Gemini*")
