import google.generativeai as genai

# Set your Google API key
genai.configure(api_key="") 

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")


def generate_mcq_from_text(text, num_questions=10):
    prompt = f"""
    Generate {num_questions} multiple-choice questions (MCQs) based on the following text.
    Each question should have 4 options (1 correct, 3 incorrect).
    Provide the correct answer after each question.
    Format:
    Question [number]: [Question text]
    A) [Option A]
    B) [Option B]
    C) [Option C]
    D) [Option D]
    Correct Answer: [Correct option] [Option text]
    Don't include ** ** or any other formatting.
    Text: {text}
    """

    response = model.generate_content(prompt)
    return response.text

# Example usage
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

pdf_text = extract_text_from_pdf("sample.pdf")
print(pdf_text[:500000]) 
mcqs = generate_mcq_from_text(pdf_text)
print(mcqs)
