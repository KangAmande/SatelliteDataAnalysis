from flask import Flask, request, jsonify
from transformers import pipeline
import torch
from philosopherModel import PhilosopherModel, answer_question_from_speech
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the pre-trained model
qa_model = pipeline("question-answering")

# Load your custom philosopher model
philosopher_model = PhilosopherModel()
philosopher_model.load_state_dict(torch.load('philosopher_model.pt'))
philosopher_model.eval()

@app.route("/answer_question", methods=["POST"])
def get_answer():
    data = request.get_json()

    pdf_path = "TestBook.pdf"
    answer_question_from_speech(pdf_path)
    question = data["question"]
    context = data["context"]

    # Use the pre-trained model for question answering
    answer = qa_model(question=question, context=context)
    return jsonify(answer)

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)