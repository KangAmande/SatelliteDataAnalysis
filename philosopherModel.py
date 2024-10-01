import torch
import torch.nn as nn
import torch.optim as optim
import fitz  # PyMuPDF
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import pipeline
import json

# Download necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class PhilosopherModel(nn.Module):
    def __init__(self):
        super(PhilosopherModel, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def train_and_save_model():
    # Dummy data for illustration
    inputs = torch.randn(100, 10)  # 100 samples, each with 10 features
    targets = torch.randn(100, 1)  # 100 target values

    # Create an instance of the model
    model = PhilosopherModel()

    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        for input, target in zip(inputs, targets):
            # Forward pass
            output = model(input)
            loss = criterion(output, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model's state dictionary
    torch.save(model.state_dict(), 'philosopher_model.pt')
    print("Model saved to philosopher_model.pt")

    # Preprocess and save the text
    pdf_path = "HolyBook.pdf"
    text = extract_text_from_pdf(pdf_path)
    preprocessed_text = preprocess_text(text)
    with open('preprocessed_text.json', 'w') as f:
        json.dump(preprocessed_text, f)
    print("Preprocessed text saved to preprocessed_text.json")

def load_model():
    model = PhilosopherModel()
    model.load_state_dict(torch.load('philosopher_model.pt'))
    model.eval()
    return model

def load_preprocessed_text():
    with open('preprocessed_text.json', 'r') as f:
        preprocessed_text = json.load(f)
    return preprocessed_text

def answer_question(question, preprocessed_text, qa_model):
    # Use the pre-trained model for question answering
    answer = qa_model(question=question, context=preprocessed_text)
    return answer

if __name__ == "__main__":
    train_and_save_model()