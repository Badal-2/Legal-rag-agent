import os
from PyPDF2 import PdfReader
from typing import Dict, List



def extract_text_from_pdf(pdf_path: str) -> Dict:
    try:
        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "error": f"File not found: {pdf_path}"
            }
        if not pdf_path.lower().endswith('.pdf'):
            return {
                "success": False,
                "error": "File must be a PDF (.pdf extension)"
            }
        
        reader = PdfReader(pdf_path)
        
        # Get number of pages
        num_pages = len(reader.pages)
        
        # Extract text from all pages
        all_text = []
        page_texts = {}  # Store text per page
        
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            
            # Clean the text (remove extra whitespace)
            text = " ".join(text.split())
            
            all_text.append(text)
            page_texts[page_num + 1] = text  # Page numbers start from 1
        
        # Combine all text
        combined_text = " ".join(all_text)
        
        # Get metadata
        metadata = {
            "filename": os.path.basename(pdf_path),
            "num_pages": num_pages,
            "total_characters": len(combined_text),
            "total_words": len(combined_text.split())
        }
        
        return {
            "success": True,
            "text": combined_text,
            "page_texts": page_texts,
            "metadata": metadata
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error extracting PDF: {str(e)}"
        }


def get_pdf_metadata(pdf_path: str) -> Dict:    #👉Get metadata from PDF without extracting full text.
    try:
        if not os.path.exists(pdf_path):
            return {"success": False, "error": "File not found"}
        
        reader = PdfReader(pdf_path)
        
        metadata = {
            "success": True,
            "filename": os.path.basename(pdf_path),
            "num_pages": len(reader.pages),
            "pdf_metadata": reader.metadata if reader.metadata else {}
        }
        
        return metadata
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading metadata: {str(e)}"
        }