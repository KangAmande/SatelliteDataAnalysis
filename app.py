from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load the pre-trained model

