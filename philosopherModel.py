import fitz # PyMuPDF
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
import torch

# Extract text from the pdf
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Pre-process the text 
def preprocess_text(text):
    # Implement tokenization, stop word removal
    
    return text

# Load pre trained QA model
qa_model = pipeline("question-answering")

# Integrate speech recognition
recognizer = sr.Recognizer()

# Integrate text to speech
engine = pyttsx3.init()

# Build the application
def answer_question_from_speech(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(text)

    while True:
        with sr.Microphone() as source:
            print("Ask a question:")
            audio = recognizer.listen(source)
            try:
                question = recognizer.recognize_google(audio)
                print(f"Question: {question}")

                if question == "exit":
                    break
                
                # Get answer from QA model
                answer = qa_model(question=question, context=preprocessed_text)
                print(f"Answer: {answer['answer']}")

                # Convert answer to speech
                engine.say(answer['answer'])
                engine.runAndWait()
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))

# Run the application
pdf_path = "HolyBook.pdf"
answer_question_from_speech(pdf_path)

# Save the model
torch.save(qa_model.state_dict(), "philosopher_model.pt")


