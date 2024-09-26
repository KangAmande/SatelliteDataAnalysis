from flask import Flask, request, jsonify
from transformers import pipeline
from philosopherModel import answer_question

app = Flask(__name__)

# Load the pre-trained model
qa_model = pipeline("question-answering")

@app.route("/answer_question", methods=["POST"])
def answer_question():
    data = request.get_json()
    question = data["question"]
    context = data["context"]

    answer = qa_model(question=question, context=context)
    return jsonify(answer)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

