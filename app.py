from flask import Flask, request, jsonify
from flask_cors import CORS
from philosopherModel import load_preprocessed_text, load_model, answer_question
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load the preprocessed text and model once when the application starts
preprocessed_text = load_preprocessed_text()
qa_model = pipeline("question-answering")

@app.route("/answer_question", methods=["POST"])
def get_answer():
    data = request.get_json()
    question = data["question"]

    # Use the custom philosopher model to answer the question
    answer = answer_question(question, preprocessed_text, qa_model)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)