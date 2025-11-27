import os
import sys
from fastapi import FastAPI

# Make project root importable
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from text_models.bigram_model import BigramModel
from rl_posttrain import format_answer

app = FastAPI(
    title="GenAI Assignment 5 – RL Post-Training API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Initialize the text model
text_model = BigramModel("this is a simple training text for this class assignment")

@app.get("/")
def root():
    return {"status": "RL API is running"}

@app.get("/text-answer")
def text_answer(prompt: str = "this"):
    """
    Generate a response from the BigramModel
    and wrap it using the RL formatting function.
    """
    raw = text_model.generate(prompt)
    formatted = format_answer(raw)
    return {"raw": raw, "formatted": formatted}
