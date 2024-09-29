from flask import Flask, request, jsonify
from flask_cors import CORS
from philosopherModel import answer_question

app = Flask(__name__)
CORS(app)

@app.route("/answer_question", methods=["POST"])
def get_answer():
    data = request.get_json()
    question = data["question"]

    # Use the custom philosopher model to answer the question
    answer = answer_question(question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    print("Server running at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)