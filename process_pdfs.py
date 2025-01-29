import PyPDF2

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

# Example usage
if __name__ == "__main__":
    pdf_path = 'sample.pdf'  # Replace with your PDF file path
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text) 