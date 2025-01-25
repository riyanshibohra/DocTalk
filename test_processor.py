import os
from document_processor import DocumentProcessor

def test_pdf_processing():
    # Initialize processor
    processor = DocumentProcessor(chunk_size=500)
    
    # Test directory with sample PDFs
    test_dir = "test_pdfs"
    
    # Create test directory if it doesn't exist
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created test directory: {test_dir}")
        print("Please add some PDF files to the test_pdfs directory")
        return
    
    # Process each PDF in the test directory
    for filename in os.listdir(test_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(test_dir, filename)
            print(f"\nProcessing: {filename}")
            
            try:
                result = processor.process_document(pdf_path)
                
                print("Metadata:")
                for key, value in result['metadata'].items():
                    print(f"  {key}: {value}")
                
                print(f"\nNumber of chunks: {len(result['chunks'])}")
                print("First chunk preview:")
                print(result['chunks'][0][:200] + "...")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    test_pdf_processing() 