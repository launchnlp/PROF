import re
import pytesseract
from pdf2image import convert_from_path
from docx import Document


def read_docx(file_path: str) -> str:
    '''
        Returning the text from a docx file
    '''
    document = Document(file_path)
    final_string = ''
    for para in document.paragraphs:
        final_string += para.text + '\n'
    return final_string

def doc_cleaner(text: str) -> str:
    '''
        Cleaning the text from a docx file
    '''
    
    # remove all occurrences of the pattern Word Count: followed by a number
    text = re.sub(r'Word Count: \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Word Count: \d+ words', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Word Count - \d+ words', '', text, flags=re.IGNORECASE)

    # remove all occurrences of the form r'(\d+ words)'
    text = re.sub(r'\(\d+ words\)', '', text)

    # remove all the leading and trailing whitespaces
    text = text.strip()

    return text

def extract_text_from_pdf_with_formatting(pdf_path):
    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=300)  # Higher DPI results in better OCR accuracy

    # Initialize a variable to store the extracted text
    extracted_text = []

    # Process each page image with pytesseract
    for image in images:
        # Use pytesseract to do OCR on the image
        text = pytesseract.image_to_string(image, lang='eng', config='--psm 6')

        # remove those newline characters if they only occur once and are not followed by capitalized words
        text = re.sub(r'\n(?![A-Z])', ' ', text)

        # Append the extracted text for this page to the list
        extracted_text.append(text)

    # Join all text into a single string
    full_text = "\n\n".join(extracted_text)
    return full_text