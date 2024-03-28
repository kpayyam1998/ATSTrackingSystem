import os 
import PyPDF2
import json 
import traceback


def read_file(file):
    try:
        # Initialize PDF Reader
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        
        # Iterate over each page in the PDF
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        
        return text
    except Exception as e:
        # Raise a more specific exception message
        raise Exception(f"Error while loading or processing the PDF file: {e}")
    

